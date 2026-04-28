# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""NPU (Ascend ACL) runtime binding utilities for stream capture status detection.

Provides ``capture_status_npu`` and ``is_capturing_npu`` as NPU equivalents
of the CUDA ``cudaStreamGetCaptureInfo`` based functions used in the
breakable CUDA graph module.
"""

import ctypes
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ACL capture status enum  (aclmdlRICaptureStatus)
# ---------------------------------------------------------------------------
# These values are inferred from:
#   - torch_npu C++ code:  c10_npu::CaptureStatus::Active
#   - vllm-ascend errors:  ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE
#   - CUDA convention:     cudaStreamCaptureStatusNone=0, Active=1
#
# **Must verify** against the ACL header on the target CANN installation:
#   grep -rn "aclmdlRICaptureStatus" /usr/local/Ascend/ascend-toolkit/*/include/
# ---------------------------------------------------------------------------
ACL_MODEL_RI_CAPTURE_STATUS_NONE = 0
ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE = 1

# ---------------------------------------------------------------------------
# Lazy-loaded ACL library handle
# ---------------------------------------------------------------------------
_acl_lib = None


def _load_acl_lib():
    """Try to load the ACL shared library. Returns the ctypes CDLL or None."""
    global _acl_lib
    if _acl_lib is not None:
        return _acl_lib

    candidates = [
        "libascendcl.so",
    ]
    for name in candidates:
        try:
            _acl_lib = ctypes.CDLL(name)
            logger.debug("Loaded ACL library: %s", name)
            return _acl_lib
        except OSError:
            continue

    _acl_lib = False
    logger.warning(
        "Could not load ACL library for NPU capture status detection. "
        "Tried: %s",
        ", ".join(candidates),
    )
    return None


def _ensure_acl_lib():
    lib = _load_acl_lib()
    if lib is None or lib is False:
        raise RuntimeError(
            "ACL library not available. "
            "Ensure CANN toolkit is installed and LD_LIBRARY_PATH is set."
        )
    return lib


# ---------------------------------------------------------------------------
# ACL API wrappers
# ---------------------------------------------------------------------------


def _aclmdlRICaptureGetInfo(stream_ptr: int) -> int:
    """Call ``aclmdlRICaptureGetInfo`` to query capture status of a stream.

    Equivalent to CUDA's ``cudaStreamGetCaptureInfo(stream_ptr)``.

    Returns the ``aclmdlRICaptureStatus`` integer value.
    """
    lib = _ensure_acl_lib()

    # int aclmdlRICaptureGetInfo(aclrtStream stream,
    #                            aclmdlRICaptureStatus *status,
    #                            aclmdlRI *modelRI)
    fn = lib.aclmdlRICaptureGetInfo
    fn.restype = ctypes.c_int
    fn.argtypes = [
        ctypes.c_void_p,  # aclrtStream stream
        ctypes.POINTER(ctypes.c_int),  # aclmdlRICaptureStatus *status
        ctypes.POINTER(ctypes.c_void_p),  # aclmdlRI *modelRI
    ]

    status = ctypes.c_int(0)
    ri = ctypes.c_void_p(0)
    ret = fn(ctypes.c_void_p(stream_ptr), ctypes.byref(status), ctypes.byref(ri))
    if ret != 0:
        raise RuntimeError(
            f"aclmdlRICaptureGetInfo failed with error code {ret} "
            f"for stream {stream_ptr:#x}"
        )
    return status.value


def capture_status_npu(stream_ptr: int) -> int:
    """Return the capture status of the given NPU stream.

    Returns one of the ``ACL_MODEL_RI_CAPTURE_STATUS_*`` constants.
    Equivalent to the CUDA ``_capture_status`` function.
    """
    return _aclmdlRICaptureGetInfo(stream_ptr)


def is_capturing_npu(stream_ptr: int) -> bool:
    """Return ``True`` if the given NPU stream is actively capturing.

    Equivalent to the CUDA ``_is_capturing`` function.
    """
    return capture_status_npu(stream_ptr) == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE
