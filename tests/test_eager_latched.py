"""Exercise batched latching when vLLM falls back to eager execution."""
import unittest
from unittest.mock import patch

import torch

from river_vllm.eager import EagerDispatch
from river_vllm.runtime import DEFAULT_DECODE_GRAPH


class Layer:
    def __init__(self, after):
        self.after = torch.tensor(after, dtype=torch.bfloat16)
        self.calls = []

    def __call__(self, positions, hidden, residual):
        self.calls.append('bf16')
        return self.after.to(hidden.device).index_select(0, positions), torch.zeros_like(hidden)

    def _forward_quant(self, positions, hidden, residual):
        self.calls.append('w4')
        return hidden, residual


class Body:
    def __init__(self):
        self.end_layer = 4
        self.layers = [Layer([[1, 0], [0, 1], [0, 1]]),
                       Layer([[0, 1], [0, 1], [1, 0]]),
                       Layer([[0, 1], [0, 1], [0, 1]]),
                       Layer([[0, 1], [0, 1], [1, 0]])]
        self.options = dict(threshold=0.7, prefill_policy='bf16')
        self.norm = self._norm_for(self.end_layer)

    def embed_input_ids(self, ids):
        return torch.ones((len(ids), 1), dtype=torch.bfloat16) * torch.tensor([1, 0], dtype=torch.bfloat16)

    def _norm_for(self, depth):
        return lambda hidden, residual: (torch.full_like(hidden, depth), residual)

    def forward(self, *_):
        raise AssertionError('Expected the phase-aware dispatch')


class EagerLatchTest(unittest.TestCase):
    def setUp(self):
        self.body = Body()
        self.dispatch = EagerDispatch(self.body, decode_graph=DEFAULT_DECODE_GRAPH)

    def test_distinct_crossings_trigger_one_shared_exit_and_reset_next_step(self):
        rows = torch.arange(2)
        output = self.dispatch.decode(rows, self.body.embed_input_ids(rows))
        self.assertEqual(output.tolist(), [[2, 2], [2, 2]])
        self.assertEqual([layer.calls for layer in self.body.layers],
                         [['bf16'], ['bf16'], ['w4'], ['w4']])

        never = torch.tensor([2])
        output = self.dispatch.decode(never, self.body.embed_input_ids(never))
        self.assertEqual(output.tolist(), [[4, 4]])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
    def test_cuda_eager_uses_the_graph_reduction_at_the_threshold(self):
        # With this BF16 vector, the ATen and Triton reduction orders round to
        # norms on opposite sides of a BF16 rounding boundary.
        hidden = torch.zeros((1, 2048), dtype=torch.bfloat16, device='cuda')
        hidden[0, :4] = torch.tensor([1., .0625, .0625, .00390625])
        hidden[0, 1024:1028] = 2 ** -12
        before = torch.zeros_like(hidden)
        before[0, 0], before[0, 4] = .50390625, .86328125
        self.body.layers = [Layer(hidden.tolist()) for _ in range(self.body.end_layer)]
        self.body.options['threshold'] = .5
        output = self.dispatch.decode(torch.tensor([0], device='cuda'), before)
        self.assertTrue((output == 1).all().item())
        self.assertEqual(self.body.layers[1].calls, ['w4'])

    def test_prefill_does_not_vote_and_uses_its_own_norm(self):
        def run_mixed(layer, positions, hidden, residual, row_sets):
            self.assertEqual([rows.tolist() for rows in row_sets], [[2], [0, 1]])
            layer.calls.append('mixed')
            return hidden, residual

        rows = torch.arange(3)
        with patch('river_vllm.eager.mixed_layer', side_effect=run_mixed):
            output, depth = self.dispatch.pack(rows, rows, (0, 1, 2), (1, 2, 3), (False, False, True))
        self.assertEqual(depth, 2)
        self.assertEqual(output.tolist(), [[2, 2], [2, 2], [4, 4]])
        self.assertEqual([layer.calls for layer in self.body.layers],
                         [['bf16'], ['bf16'], ['mixed'], ['mixed']])


if __name__ == '__main__':
    unittest.main()
