# Copyright 2025 SGLang Team
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
"""NPU breakable graph runner.

Applies monkey patches to make BreakableCudaGraphRunner work on Ascend NPU.
Patches are applied in __init__ before calling super().__init__(), so that
the core breakable_cuda_graph.py requires zero changes.
"""

from __future__ import annotations

import ctypes
import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.model_executor.breakable_cuda_graph_runner import (
    BreakableCudaGraphRunner,
)
from sglang.srt.utils import is_npu

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ACL capture status query via ctypes
# ---------------------------------------------------------------------------

_acl_lib = None
_acl_fn = None


def _get_acl_capture_fn():
    """Lazy-load aclmdlRICaptureGetInfo from libascendcl.so."""
    global _acl_lib, _acl_fn
    if _acl_fn is not None:
        return _acl_fn

    import os

    candidates = ["libascendcl.so", "libascendcl.so.1"]
    cann_home = os.environ.get("ASCEND_HOME_PATH", "")
    if cann_home:
        candidates.insert(0, f"{cann_home}/lib64/libascendcl.so")

    for name in candidates:
        try:
            _acl_lib = ctypes.CDLL(name)
            break
        except OSError:
            continue

    if _acl_lib is None:
        raise RuntimeError(
            "Could not load libascendcl.so for BCG NPU adaptation. "
            "Check CANN installation and LD_LIBRARY_PATH."
        )

    fn = _acl_lib.aclmdlRICaptureGetInfo
    fn.restype = ctypes.c_int
    fn.argtypes = [
        ctypes.c_void_p,  # aclrtStream stream
        ctypes.POINTER(ctypes.c_int),  # aclmdlRICaptureStatus *status
        ctypes.POINTER(ctypes.c_void_p),  # aclmdlRI *modelRI
    ]
    _acl_fn = fn
    return fn


def _npu_capture_status(stream_ptr: int) -> int:
    """Query NPU stream capture status via ACL.

    Returns 0 (NONE) or 1 (ACTIVE), matching CUDA's cudaStreamCaptureStatus enum.
    """
    fn = _get_acl_capture_fn()
    status = ctypes.c_int(-1)
    ri = ctypes.c_void_p(0)
    fn(ctypes.c_void_p(stream_ptr), ctypes.byref(status), ctypes.byref(ri))
    return status.value


def _npu_is_capturing(stream_ptr: int) -> bool:
    return _npu_capture_status(stream_ptr) == 1


# ---------------------------------------------------------------------------
# Patch NPUGraph.capture_begin to accept and ignore capture_error_mode
# ---------------------------------------------------------------------------

_original_npu_capture_begin = torch.npu.NPUGraph.capture_begin


def _patched_npu_capture_begin(self, pool=None, capture_error_mode=None, **kwargs):
    return _original_npu_capture_begin(self, pool=pool)


# ---------------------------------------------------------------------------
# Apply all monkey patches
# ---------------------------------------------------------------------------

_patches_applied = False


def _apply_bcg_npu_patches():
    """Apply all monkey patches needed for BCG on NPU.

    Must be called before BreakableCudaGraphRunner uses BCG capture.
    """
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    # --- Patch 1: torch.cuda aliases (same as eagle_draft_npu_graph_runner) ---
    torch.cuda.CUDAGraph = torch.npu.NPUGraph
    torch.cuda.synchronize = torch.npu.synchronize
    torch.cuda.graph = torch.npu.graph
    torch.cuda.stream = torch.npu.stream
    torch.cuda.Stream = torch.npu.Stream
    torch.cuda.current_stream = torch.npu.current_stream
    logger.info("[BCG NPU] Applied torch.cuda -> torch.npu aliases")

    # --- Patch 2: stream.cuda_stream property alias ---
    if not hasattr(torch.npu.Stream, "cuda_stream"):
        torch.npu.Stream.cuda_stream = property(
            lambda self: self.npu_stream,
            doc="Alias for npu_stream, used by BCG fork/join detection.",
        )
        logger.info("[BCG NPU] Added torch.npu.Stream.cuda_stream property alias")

    # --- Patch 3: replace capture status functions with ACL versions ---
    import sglang.srt.model_executor.breakable_cuda_graph.breakable_cuda_graph as bcg

    bcg._capture_status = _npu_capture_status
    bcg._is_capturing = _npu_is_capturing
    logger.info("[BCG NPU] Replaced _capture_status/_is_capturing with ACL versions")

    # --- Patch 4: NPUGraph.capture_begin accepts capture_error_mode ---
    torch.npu.NPUGraph.capture_begin = _patched_npu_capture_begin
    logger.info("[BCG NPU] Patched NPUGraph.capture_begin to accept capture_error_mode")


# ---------------------------------------------------------------------------
# NPU Breakable Graph Runner
# ---------------------------------------------------------------------------


class BreakableNpuGraphRunner(BreakableCudaGraphRunner):
    """Breakable graph runner for Ascend NPU.

    Applies NPU monkey patches in __init__ before calling
    super().__init__(), which triggers graph capture.
    """

    def __init__(self, model_runner: ModelRunner):
        _apply_bcg_npu_patches()
        super().__init__(model_runner)
