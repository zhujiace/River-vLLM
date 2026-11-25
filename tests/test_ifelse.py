"""GPU regression checks for the optional native IF/ELSE decoder."""

import unittest

import torch

from river_vllm.ifelse import IfElseDecode
from river_vllm.ifelse_bridge import conditional_capture, load_extension


class Layer:
    def __init__(self, index):
        self.index = index

    def run(self, hidden, residual, value):
        if residual is None:
            residual = torch.zeros_like(hidden)
        residual.add_(hidden)
        output = torch.zeros_like(hidden)
        output[self.index].fill_(value)
        return output, residual

    def __call__(self, positions, hidden, residual):
        return self.run(hidden, residual, 1)

    def _forward_quant(self, positions, hidden, residual):
        return self.run(hidden, residual, 4)


class Body:
    end_layer = 16
    layers = [Layer(index) for index in range(end_layer)]

    def embed_input_ids(self, ids):
        return ids.clone()


class ToyDecode(IfElseDecode):
    def __init__(self, target, pool):
        super().__init__(Body(), 0.7, target, 1e-5, pool=pool)
        self.target = target

    def check_predicates(self, snapshot, before, effective_depth, selected, exits, continues):
        judged = effective_depth - 1
        matched = self.target.eq(judged) & snapshot.sum().eq(judged)
        selected.copy_(torch.where(matched, effective_depth, 0))
        exits.copy_(matched)
        continues.copy_(~matched)
        return snapshot

    def normalize(self, hidden, residual, selected_depth):
        return hidden + residual


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class IfElseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_extension()

    def test_nested_device_branches_and_parent_join(self):
        from cuda.bindings import driver as cuda

        pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        predicate = torch.zeros((), dtype=torch.bool, device="cuda")
        nested = torch.zeros_like(predicate)
        before = torch.arange(1024, device="cuda")
        workspace, output = torch.empty_like(before), torch.empty_like(before)
        side = torch.cuda.Stream()
        ready, written = torch.cuda.Event(), torch.cuda.Event()

        with torch.cuda.graph(graph, pool=pool), conditional_capture(pool) as dispatch:
            main = torch.cuda.current_stream()
            handle = dispatch.new_handle()
            ready.record(main)
            with torch.cuda.stream(side):
                side.wait_event(ready)
                dispatch.set_handle(handle, predicate)
                written.record(side)
            first = before + 1
            main.wait_event(written)

            def continue_branch():
                inner = dispatch.new_handle()
                dispatch.set_handle(inner, nested)
                dispatch.capture_if_else(inner, lambda: workspace.copy_(first + 5),
                                         lambda: workspace.copy_(first * 2))

            dispatch.capture_if_else(handle, lambda: workspace.copy_(first + 2), continue_branch)
            output.copy_(workspace - 1)

        raw = cuda.CUgraph(graph.raw_cuda_graph())
        error, _, count = cuda.cuGraphGetNodes(raw, 0)
        self.assertEqual(error, cuda.CUresult.CUDA_SUCCESS)
        error, nodes, _ = cuda.cuGraphGetNodes(raw, count)
        self.assertEqual(error, cuda.CUresult.CUDA_SUCCESS)
        types = [cuda.cuGraphNodeGetType(node) for node in nodes]
        self.assertTrue(all(err == cuda.CUresult.CUDA_SUCCESS for err, _ in types))
        self.assertEqual(sum(t == cuda.CUgraphNodeType.CU_GRAPH_NODE_TYPE_CONDITIONAL for _, t in types), 1)

        for pred, inner in ((False, False), (True, False), (False, True), (False, False)):
            predicate.fill_(pred)
            nested.fill_(inner)
            graph.replay()
            expected = before + 2 if pred else (before + 5 if inner else 2 * before + 1)
            torch.testing.assert_close(output, expected, atol=0, rtol=0)

    def test_all_delayed_depths_with_inplace_residual(self):
        ids = torch.zeros(16, device="cuda")
        target = torch.zeros((), dtype=torch.long, device="cuda")
        pool = torch.cuda.graph_pool_handle()
        decoder = ToyDecode(target, pool)
        stream = decoder.stages[5][0]
        decoder.exclude_capture_stream(stream)
        self.assertNotIn(stream.cuda_stream, {stage[0].cuda_stream for stage in decoder.stages})
        output, depth = torch.empty_like(ids), torch.empty_like(target)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool, stream=stream):
            decoder.forward(ids, ids, output, depth)
        for judged in (0, *range(16, 0, -1), 1, 6, 0):
            with self.subTest(judged=judged):
                effective = min(judged + 1, 16) if judged else 16
                target.fill_(judged)
                graph.replay()
                self.assertEqual(depth.item(), effective)
                self.assertEqual(output.tolist(), [1] * effective + [4] * (16 - effective))


if __name__ == "__main__":
    unittest.main()
