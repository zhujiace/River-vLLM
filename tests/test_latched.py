"""Verify the latched GPU route across layers and successive graph replays."""
import unittest

import torch

from river_vllm.conditional_decode import normalize_for_cosine
from river_vllm.latched import cosine_scores, update_latched


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required')
class LatchedGateTest(unittest.TestCase):
    def test_fused_cosine_matches_bf16_intermediates(self):
        generator = torch.Generator(device='cuda').manual_seed(2793)
        epsilon = torch.scalar_tensor(1e-8, device='cuda')
        for rows, width in ((1, 2048), (7, 2048), (64, 2048), (5, 192)):
            hidden, residual, previous = torch.randn((3, rows, width), device='cuda', generator=generator).bfloat16()
            hidden[-1].zero_()
            residual[-1].zero_()
            before = normalize_for_cosine(previous, epsilon)
            expected = normalize_for_cosine(hidden + residual, epsilon)
            normalized, scores = cosine_scores(hidden, residual, before)
            torch.testing.assert_close(normalized, expected, atol=0, rtol=0)
            torch.testing.assert_close(scores, (expected * before).sum(-1), atol=0, rtol=0)

    def test_replay_discards_prior_crossings_and_ignores_padding(self):
        slots = torch.tensor([9, -1, 3, 7, -1, 0, 1], device='cuda')
        scores = torch.empty((3, len(slots)), dtype=torch.bfloat16, device='cuda')
        seen = torch.empty_like(slots, dtype=torch.bool)
        threshold = torch.tensor(0.7, dtype=scores.dtype).item()
        depths = torch.empty(3, device='cuda', dtype=torch.long)
        exits = torch.empty_like(depths, dtype=torch.bool)

        def route():
            for index in range(len(scores)):
                update_latched(scores[index], slots, seen, threshold, index + 1,
                               depths[index], exits[index], reset=index == 0)

        route()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            route()
        for active, data, expected in [
            ([9, -1, 3, 7, -1, 0, 1],
             [[.8, .1, .7, .9, .1, .2, .1], [.1, .1, .8, .2, .1, .9, .1], [.2, .1, .3, .2, .1, .2, .9]],
             [0, 0, 3]),
            ([9, 2, -1, -1, -1, -1, -1],
             [[.1, .7, .9, .9, .9, .9, .9], [.1, .9, .1, .1, .1, .1, .1], [.1, .1, .1, .1, .1, .1, .1]],
             [0, 0, 0]),
            ([9, 2, -1, -1, -1, -1, -1],
             [[.9, .9, .1, .1, .1, .1, .1]] * 3,
             [1, 2, 3]),
        ]:
            with self.subTest(expected=expected):
                slots.copy_(torch.tensor(active))
                scores.copy_(torch.tensor(data))
                graph.replay()
                self.assertEqual(depths.tolist(), expected)
                self.assertEqual(exits.tolist(), [depth > 0 for depth in expected])


if __name__ == '__main__':
    unittest.main()
