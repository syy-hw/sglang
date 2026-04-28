"""
Minimal staged verification for NPU capture status detection.

Run on an Ascend NPU machine. Each stage verifies one prerequisite.
Stops at the first failure so you know exactly what's missing.

Usage:
    python test_npu_capture_status.py
"""

import ctypes
import sys
import traceback


def stage1_load_library():
    """Stage 1: Can we load the ACL shared library?"""
    print("=" * 60)
    print("Stage 1: Load ACL shared library")
    print("=" * 60)

    candidates = [
        "libascendcl.so",
        "libascendcl.so.1",
    ]

    import os
    cann_home = os.environ.get("ASCEND_HOME_PATH", "")
    if cann_home:
        candidates.insert(0, f"{cann_home}/lib64/libascendcl.so")

    for name in candidates:
        try:
            lib = ctypes.CDLL(name)
            print(f"  [PASS] Loaded: {name}")
            return lib
        except OSError as e:
            print(f"  [SKIP] {name}: {e}")

    print("  [FAIL] Could not load any ACL library")
    print("  Hint: Check CANN installation and LD_LIBRARY_PATH")
    return None


def stage2_resolve_symbol(lib):
    """Stage 2: Does the library export aclmdlRICaptureGetInfo?"""
    print()
    print("=" * 60)
    print("Stage 2: Resolve aclmdlRICaptureGetInfo symbol")
    print("=" * 60)

    try:
        fn = lib.aclmdlRICaptureGetInfo
        print("  [PASS] Symbol resolved: aclmdlRICaptureGetInfo")
        return fn
    except AttributeError:
        print("  [FAIL] Symbol not found: aclmdlRICaptureGetInfo")
        print("  Hint: This is an experimental API. Your CANN version may not")
        print("        expose it, or it may have a different name.")
        print()
        print("  Searching for capture-related symbols...")
        try:
            import subprocess
            import ctypes.util
            path = ctypes.util.find_library("ascendcl")
            if path:
                result = subprocess.run(
                    ["nm", "-D", path],
                    capture_output=True, text=True
                )
                found = False
                for line in result.stdout.splitlines():
                    if "capture" in line.lower() or "Capture" in line:
                        print(f"    {line.strip()}")
                        found = True
                if not found:
                    print("    (no capture-related symbols found)")
        except Exception:
            pass
        return None


def stage3_call_on_non_capturing_stream(fn):
    """Stage 3: Can we call the API on a non-capturing stream?

    This verifies the calling convention (arg types, return type) is correct.
    """
    print()
    print("=" * 60)
    print("Stage 3: Call API on non-capturing stream")
    print("=" * 60)

    fn.restype = ctypes.c_int
    fn.argtypes = [
        ctypes.c_void_p,                     # aclrtStream stream
        ctypes.POINTER(ctypes.c_int),        # aclmdlRICaptureStatus *status
        ctypes.POINTER(ctypes.c_void_p),     # aclmdlRI *modelRI
    ]

    stream_ptr = 0
    try:
        import torch
        import torch_npu  # noqa: F401
        stream = torch.npu.Stream()
        stream_ptr = stream.npu_stream
        print(f"  Created NPU stream, ptr = {stream_ptr:#x}")
    except ImportError:
        print("  [INFO] torch_npu not available, trying with NULL stream (0)")
    except Exception as e:
        print(f"  [INFO] Cannot create NPU stream: {e}")
        print("  Trying with NULL stream (0)...")

    status = ctypes.c_int(-1)
    ri = ctypes.c_void_p(0)
    ret = fn(ctypes.c_void_p(stream_ptr), ctypes.byref(status), ctypes.byref(ri))

    if ret != 0:
        print(f"  [FAIL] aclmdlRICaptureGetInfo returned error code: {ret}")
        print("  Hint: The calling convention may be wrong, or ACL needs")
        print("        initialization (aclInit) before this call.")
        return None

    print(f"  [PASS] API call succeeded, status = {status.value}")
    print(f"         ri handle = {ri.value or 0:#x}")
    return status.value


def stage4_verify_enum_values(status_non_capturing):
    """Stage 4: Verify enum value assumptions."""
    print()
    print("=" * 60)
    print("Stage 4: Verify enum values")
    print("=" * 60)

    NONE = 0
    ACTIVE = 1

    print(f"  Non-capturing stream status = {status_non_capturing}")
    print(f"  Expected NONE = {NONE}")

    if status_non_capturing == NONE:
        print("  [PASS] Non-capturing status matches NONE = 0")
    else:
        print(f"  [WARN] Non-capturing status is {status_non_capturing}, not {NONE}")
        print(f"         Update ACL_MODEL_RI_CAPTURE_STATUS_NONE in npu_utils.py")
        print(f"         The actual NONE value might be {status_non_capturing}")

    print()
    print("  To verify ACTIVE value, we need a capturing stream.")
    print("  This requires torch.npu.Stream.capture_begin() support.")
    print("  Skipping for now — verify manually during BCG integration.")


def stage5_capture_status_on_capturing_stream(lib):
    """Stage 5 (optional): Verify status = ACTIVE during actual capture."""
    print()
    print("=" * 60)
    print("Stage 5 (optional): Capture status during graph capture")
    print("=" * 60)

    try:
        import torch
        import torch_npu  # noqa: F401
    except ImportError:
        print("  [SKIP] torch_npu not available")
        return

    try:
        stream = torch.npu.Stream()
        stream_ptr = stream.npu_stream

        graph = torch.npu.NPUGraph()
        graph.capture_begin(pool=None)

        fn = lib.aclmdlRICaptureGetInfo
        fn.restype = ctypes.c_int
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        status = ctypes.c_int(-1)
        ri = ctypes.c_void_p(0)
        ret = fn(ctypes.c_void_p(stream_ptr), ctypes.byref(status), ctypes.byref(ri))

        graph.capture_end()

        if ret != 0:
            print(f"  [FAIL] API call during capture returned: {ret}")
            return

        print(f"  Capturing stream status = {status.value}")

        if status.value == 1:
            print("  [PASS] ACTIVE = 1 confirmed!")
        else:
            print(f"  [WARN] Status during capture = {status.value}")
            print("         Expected ACTIVE = 1. Update npu_utils.py accordingly.")

    except AttributeError as e:
        print(f"  [SKIP] NPUGraph.capture_begin not available: {e}")
    except Exception as e:
        print(f"  [FAIL] {e}")
        traceback.print_exc()


def main():
    print("NPU Capture Status Detection — Staged Verification")
    print(f"Python: {sys.version}")
    print()

    lib = stage1_load_library()
    if lib is None:
        print("\nStopped at Stage 1. Fix library loading before proceeding.")
        return

    fn = stage2_resolve_symbol(lib)
    if fn is None:
        print("\nStopped at Stage 2. The symbol doesn't exist in your CANN.")
        print("Check CANN version or look for alternative API names.")
        return

    status = stage3_call_on_non_capturing_stream(fn)
    if status is None:
        print("\nStopped at Stage 3. API call failed.")
        return

    stage4_verify_enum_values(status)
    stage5_capture_status_on_capturing_stream(lib)

    print()
    print("=" * 60)
    print("Verification complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
