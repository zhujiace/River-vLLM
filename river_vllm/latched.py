"""Run a homogeneous W4 tail once every live decode row has crossed the threshold."""

from unittest.mock import patch

import torch
import triton
import triton.language as tl
from vllm.forward_context import get_forward_context
from vllm.utils import torch_utils

from .conditional_decode import DirectTailDecode, normalize_for_cosine
from .ifelse_bridge import conditional_capture


@triton.jit
def _cosine(hidden, residual, before, after, scores, hidden_stride, residual_stride, before_stride,
            WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < WIDTH
    h = tl.load(hidden + row * hidden_stride + col, valid, other=0).to(tl.float32)
    r = tl.load(residual + row * residual_stride + col, valid, other=0).to(tl.float32)
    # Match the BF16 rounding at each intermediate tensor in the reference gate.
    x = (h + r).to(tl.bfloat16).to(tl.float32)
    norm = tl.sqrt_rn(tl.sum(x * x, axis=0)).to(tl.bfloat16).to(tl.float32)
    epsilon = 1.0011717677116394e-08  # 1e-8 rounded to BF16
    norm = tl.maximum(norm, epsilon, propagate_nan=tl.PropagateNan.ALL)
    normalized = tl.div_rn(x, norm).to(tl.bfloat16)
    previous = tl.load(before + row * before_stride + col, valid, other=0).to(tl.float32)
    products = (normalized.to(tl.float32) * previous).to(tl.bfloat16).to(tl.float32)
    tl.store(after + row * WIDTH + col, normalized, valid)
    tl.store(scores + row, tl.sum(products, axis=0))


def cosine_scores(hidden, residual, before):
    rows, width = hidden.shape
    after = hidden.new_empty((rows, width))
    scores = hidden.new_empty(rows)
    _cosine[(rows,)](hidden, residual, before, after, scores,
                     hidden.stride(0), residual.stride(0), before.stride(0),
                     width, triton.next_power_of_2(width), enable_fp_fusion=False)
    return after, scores


@triton.jit(do_not_specialize=['threshold', 'depth'])
def _update_latched(scores, slots, seen, selected, exits, threshold, depth,
                    ROWS: tl.constexpr, BLOCK: tl.constexpr, RESET: tl.constexpr):
    row = tl.arange(0, BLOCK)
    valid = tl.load(slots + row, row < ROWS, other=-1) >= 0
    scores = tl.load(scores + row, row < ROWS, other=0).to(tl.float32)
    passed = scores > threshold
    if not RESET:
        passed |= tl.load(seen + row, row < ROWS, other=False)
    tl.store(seen + row, valid & passed, row < ROWS)
    all_passed = tl.sum((valid & ~passed).to(tl.int32), axis=0) == 0
    tl.store(selected, tl.where(all_passed, depth, 0).to(tl.int64))
    tl.store(exits, all_passed)


def update_latched(scores, slots, seen, threshold, depth, selected, exits, *, reset):
    rows = slots.numel()
    _update_latched[(1,)](scores, slots, seen, selected, exits, threshold, depth,
                          ROWS=rows, BLOCK=triton.next_power_of_2(rows), RESET=reset, num_warps=1)


class LatchedDecode(DirectTailDecode):
    def __init__(self, body, threshold, norm_weights, norm_eps, *, pool):
        super().__init__(body, threshold, norm_weights, norm_eps)
        self.pool = pool
        self.rounded_threshold = torch.tensor(threshold, dtype=torch.bfloat16, device='cpu').item()

    def warm(self, positions, ids):
        super().warm(positions, ids)
        hidden = self.body.embed_input_ids(ids)
        before = normalize_for_cosine(hidden, self.epsilon)
        _, scores = cosine_scores(hidden, torch.zeros_like(hidden), before)
        slots = next(iter(get_forward_context().slot_mapping.values()))
        seen = torch.empty_like(slots, dtype=torch.bool)
        selected, exits = slots.new_empty(()), slots.new_empty((), dtype=torch.bool)
        for depth in (1, 2):
            update_latched(scores, slots, seen, self.rounded_threshold, depth, selected, exits, reset=depth == 1)

    def forward(self, positions, ids, output, selected_depth):
        body = self.body
        slots = next(iter(get_forward_context().slot_mapping.values()))
        seen = torch.empty_like(slots, dtype=torch.bool)
        exits = slots.new_empty((), dtype=torch.bool)

        def finish(depth, hidden, residual):
            for layer in body.layers[depth:]:
                hidden, residual = layer._forward_quant(positions, hidden, residual)
            output.copy_(self.normalize(hidden, residual, selected_depth))

        def advance(depth, hidden, residual, before):
            hidden, residual = body.layers[depth - 1](positions, hidden, residual)
            if depth == body.end_layer:
                selected_depth.fill_(depth)
                finish(depth, hidden, residual)
                return

            after, scores = cosine_scores(hidden, residual, before)
            update_latched(scores, slots, seen, self.rounded_threshold, depth, selected_depth, exits, reset=depth == 1)
            handle = dispatch.new_handle()
            dispatch.set_handle(handle, exits)
            dispatch.capture_if_else(handle, lambda: finish(depth, hidden, residual),
                                     lambda: advance(depth + 1, hidden, residual, after))

        # Child graphs share the workspace already warmed on the capture stream.
        root = torch.cuda.current_stream(ids.device)
        with patch.object(torch_utils, 'current_stream', return_value=root), conditional_capture(self.pool) as dispatch:
            hidden = body.embed_input_ids(ids)
            advance(1, hidden, None, normalize_for_cosine(hidden, self.epsilon))
