"""Tests for the breakable CUDA graph (BCG) runner on Ascend NPU.

Mirrors test_breakable_cuda_graph.py but applies NPU monkey-patches before
running the same capture/replay unit tests with torch.npu tensors.

Test classes:
- ``TestBreakableNPUGraphBasic``: core capture/replay with simple tensor ops
- ``TestCopyOutputNPU``: structured output writeback helper
- ``TestBreakGraphHelperNPU``: break_graph() convenience function
- ``TestNPUPatches``: verify NPU-specific monkey patches
"""

import unittest

import torch

from sglang.test.test_utils import CustomTestCase


def _apply_npu_patches():
    """Apply the same patches as _apply_bcg_npu_patches()."""
    from sglang.srt.hardware_backend.npu.graph_runner.breakable_npu_graph_runner import (
        _apply_bcg_npu_patches,
    )

    _apply_bcg_npu_patches()


class TestBreakableNPUGraphBasic(CustomTestCase):
    """Test basic breakable graph capture and replay on Ascend NPU."""

    @classmethod
    def setUpClass(cls):
        if not (hasattr(torch, "npu") and torch.npu.is_available()):
            raise unittest.SkipTest("NPU not available")

        _apply_npu_patches()

        from sglang.srt.model_executor.breakable_cuda_graph.breakable_cuda_graph import (
            BreakableCUDAGraph,
            BreakableCUDAGraphCapture,
            eager_on_graph,
        )

        cls.BreakableCUDAGraph = BreakableCUDAGraph
        cls.BreakableCUDAGraphCapture = BreakableCUDAGraphCapture
        cls.eager_on_graph = staticmethod(eager_on_graph)
        cls.device = torch.device("npu:0")

    def test_no_break_capture_replay(self):
        """Capture and replay without any graph breaks."""
        x = torch.zeros(4, device=self.device)
        y = torch.zeros(4, device=self.device)

        graph = self.BreakableCUDAGraph()
        stream = torch.npu.Stream(self.device)
        with self.BreakableCUDAGraphCapture(graph, stream=stream):
            y.copy_(x + 1.0)

        x.fill_(5.0)
        graph.replay()
        torch.npu.synchronize()
        self.assertTrue(torch.allclose(y, torch.full((4,), 6.0, device=self.device)))

    def test_single_break(self):
        """A single graph break should split capture into two segments."""
        x = torch.zeros(4, device=self.device)
        intermediate = torch.zeros(4, device=self.device)
        y = torch.zeros(4, device=self.device)

        @self.eager_on_graph(enable=True)
        def eager_op(src):
            return src * 2.0

        graph = self.BreakableCUDAGraph()
        stream = torch.npu.Stream(self.device)
        with self.BreakableCUDAGraphCapture(graph, stream=stream):
            intermediate.copy_(x + 1.0)
            broken = eager_op(intermediate)
            y.copy_(broken + 3.0)

        x.fill_(10.0)
        graph.replay()
        torch.npu.synchronize()
        # x=10 -> intermediate=11 -> eager: 11*2=22 -> y=22+3=25
        self.assertTrue(torch.allclose(y, torch.full((4,), 25.0, device=self.device)))

    def test_multiple_breaks(self):
        """Multiple graph breaks should produce correct chained results."""
        x = torch.zeros(4, device=self.device)
        y = torch.zeros(4, device=self.device)

        @self.eager_on_graph(enable=True)
        def add_one(src):
            return src + 1.0

        @self.eager_on_graph(enable=True)
        def double(src):
            return src * 2.0

        graph = self.BreakableCUDAGraph()
        stream = torch.npu.Stream(self.device)
        with self.BreakableCUDAGraphCapture(graph, stream=stream):
            t1 = x + 1.0
            t2 = add_one(t1)
            t3 = t2 + 1.0
            t4 = double(t3)
            y.copy_(t4)

        # Replay: x=5 -> +1=6 -> add_one=7 -> +1=8 -> double=16
        x.fill_(5.0)
        graph.replay()
        torch.npu.synchronize()
        self.assertTrue(torch.allclose(y, torch.full((4,), 16.0, device=self.device)))

    def test_eager_on_graph_disabled(self):
        """@eager_on_graph(enable=False) should be a no-op passthrough."""

        @self.eager_on_graph(enable=False)
        def my_fn(x):
            return x + 1.0

        t = torch.tensor([1.0, 2.0], device=self.device)
        result = my_fn(t)
        self.assertTrue(
            torch.allclose(result, torch.tensor([2.0, 3.0], device=self.device))
        )

    def test_eager_on_graph_outside_capture(self):
        """@eager_on_graph called outside capture should run the function directly."""

        @self.eager_on_graph(enable=True)
        def my_fn(x):
            return x + 1.0

        t = torch.tensor([1.0, 2.0], device=self.device)
        result = my_fn(t)
        self.assertTrue(
            torch.allclose(result, torch.tensor([2.0, 3.0], device=self.device))
        )

    def test_replay_updates_output(self):
        """Replay should produce different results when input buffers change."""
        x = torch.zeros(4, device=self.device)
        y = torch.zeros(4, device=self.device)

        @self.eager_on_graph(enable=True)
        def scale(src):
            return src * 3.0

        graph = self.BreakableCUDAGraph()
        stream = torch.npu.Stream(self.device)
        with self.BreakableCUDAGraphCapture(graph, stream=stream):
            t = x + 1.0
            t2 = scale(t)
            y.copy_(t2)

        # First replay: x=0 -> 0+1=1 -> 1*3=3
        graph.replay()
        torch.npu.synchronize()
        self.assertTrue(torch.allclose(y, torch.full((4,), 3.0, device=self.device)))

        # Second replay: x=10 -> 10+1=11 -> 11*3=33
        x.fill_(10.0)
        graph.replay()
        torch.npu.synchronize()
        self.assertTrue(torch.allclose(y, torch.full((4,), 33.0, device=self.device)))

    def test_segment_count_single_break(self):
        """Single break should produce exactly 2 segments."""
        x = torch.zeros(4, device=self.device)
        y = torch.zeros(4, device=self.device)

        @self.eager_on_graph(enable=True)
        def eager_op(src):
            return src * 2.0

        graph = self.BreakableCUDAGraph()
        stream = torch.npu.Stream(self.device)
        with self.BreakableCUDAGraphCapture(graph, stream=stream):
            intermediate = x + 1.0
            broken = eager_op(intermediate)
            y.copy_(broken)

        self.assertEqual(len(graph._segments), 2)
        self.assertEqual(len(graph._break_fns), 1)

    def test_segment_count_multiple_breaks(self):
        """Two breaks should produce exactly 3 segments."""
        x = torch.zeros(4, device=self.device)
        y = torch.zeros(4, device=self.device)

        @self.eager_on_graph(enable=True)
        def op1(src):
            return src + 1.0

        @self.eager_on_graph(enable=True)
        def op2(src):
            return src * 2.0

        graph = self.BreakableCUDAGraph()
        stream = torch.npu.Stream(self.device)
        with self.BreakableCUDAGraphCapture(graph, stream=stream):
            t1 = x + 1.0
            t2 = op1(t1)
            t3 = op2(t2)
            y.copy_(t3)

        self.assertEqual(len(graph._segments), 3)
        self.assertEqual(len(graph._break_fns), 2)


class TestCopyOutputNPU(CustomTestCase):
    """Test the _copy_output helper for structured output writeback on NPU."""

    @classmethod
    def setUpClass(cls):
        if not (hasattr(torch, "npu") and torch.npu.is_available()):
            raise unittest.SkipTest("NPU not available")

        _apply_npu_patches()

        from sglang.srt.model_executor.breakable_cuda_graph.breakable_cuda_graph import (
            _copy_output,
        )

        cls._copy_output = staticmethod(_copy_output)
        cls.device = torch.device("npu:0")

    def test_tensor_copy(self):
        dst = torch.zeros(4, device=self.device)
        src = torch.ones(4, device=self.device) * 5.0
        result = self._copy_output(dst, src)
        self.assertIs(result, dst)
        self.assertTrue(torch.allclose(dst, src))

    def test_dict_copy(self):
        dst = {
            "a": torch.zeros(4, device=self.device),
            "b": torch.zeros(4, device=self.device),
        }
        src = {
            "a": torch.ones(4, device=self.device),
            "b": torch.ones(4, device=self.device) * 2.0,
        }
        result = self._copy_output(dst, src)
        self.assertIs(result, dst)
        self.assertTrue(torch.allclose(dst["a"], torch.ones(4, device=self.device)))
        self.assertTrue(
            torch.allclose(dst["b"], torch.ones(4, device=self.device) * 2.0)
        )

    def test_object_copy(self):
        class FakeOutput:
            def __init__(self, t, label):
                self.tensor = t
                self.label = label

        dst = FakeOutput(torch.zeros(4, device=self.device), "old")
        src = FakeOutput(torch.ones(4, device=self.device) * 3.0, "new")
        result = self._copy_output(dst, src)
        self.assertIs(result, dst)
        self.assertTrue(
            torch.allclose(dst.tensor, torch.ones(4, device=self.device) * 3.0)
        )
        self.assertEqual(dst.label, "new")

    def test_non_tensor_fallback(self):
        result = self._copy_output(42, 99)
        self.assertEqual(result, 99)


class TestBreakGraphHelperNPU(CustomTestCase):
    """Test the break_graph() convenience function on NPU."""

    @classmethod
    def setUpClass(cls):
        if not (hasattr(torch, "npu") and torch.npu.is_available()):
            raise unittest.SkipTest("NPU not available")

        _apply_npu_patches()

        from sglang.srt.model_executor.breakable_cuda_graph.breakable_cuda_graph import (
            BreakableCUDAGraph,
            BreakableCUDAGraphCapture,
            break_graph,
        )

        cls.BreakableCUDAGraph = BreakableCUDAGraph
        cls.BreakableCUDAGraphCapture = BreakableCUDAGraphCapture
        cls.break_graph = staticmethod(break_graph)
        cls.device = torch.device("npu:0")

    def test_break_graph_inserts_segment(self):
        """break_graph() should insert a graph break on NPU."""
        x = torch.zeros(4, device=self.device)
        y = torch.zeros(4, device=self.device)

        graph = self.BreakableCUDAGraph()
        stream = torch.npu.Stream(self.device)
        with self.BreakableCUDAGraphCapture(graph, stream=stream):
            t = x + 1.0
            self.break_graph()
            y.copy_(t + 2.0)

        x.fill_(10.0)
        graph.replay()
        torch.npu.synchronize()
        # x=10 -> +1=11 -> break -> +2=13
        self.assertTrue(torch.allclose(y, torch.full((4,), 13.0, device=self.device)))


class TestNPUPatches(CustomTestCase):
    """Verify NPU-specific monkey patches are applied correctly."""

    @classmethod
    def setUpClass(cls):
        if not (hasattr(torch, "npu") and torch.npu.is_available()):
            raise unittest.SkipTest("NPU not available")

        _apply_npu_patches()

    def test_cuda_graph_is_npu_graph(self):
        """torch.cuda.CUDAGraph should be aliased to torch.npu.NPUGraph."""
        self.assertIs(torch.cuda.CUDAGraph, torch.npu.NPUGraph)

    def test_cuda_synchronize_is_npu_synchronize(self):
        """torch.cuda.synchronize should be aliased to torch.npu.synchronize."""
        self.assertIs(torch.cuda.synchronize, torch.npu.synchronize)

    def test_cuda_stream_is_npu_stream(self):
        """torch.cuda.Stream should be aliased to torch.npu.Stream."""
        self.assertIs(torch.cuda.Stream, torch.npu.Stream)

    def test_stream_cuda_stream_property(self):
        """torch.npu.Stream should have cuda_stream property aliasing npu_stream."""
        s = torch.npu.Stream()
        self.assertEqual(s.cuda_stream, s.npu_stream)

    def test_capture_status_function(self):
        """NPU capture status function should return int (0 or 1)."""
        from sglang.srt.hardware_backend.npu.graph_runner.breakable_npu_graph_runner import (
            _npu_capture_status,
        )

        s = torch.npu.current_stream()
        status = _npu_capture_status(s.npu_stream)
        self.assertIn(status, [0, 1])


if __name__ == "__main__":
    unittest.main()
