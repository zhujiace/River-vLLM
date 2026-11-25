"""Overlap the delayed exit check with BF16 and dispatch a native IF/ELSE."""

from unittest.mock import patch

import torch
import triton
import triton.language as tl
from vllm.forward_context import get_forward_context
from vllm.utils import torch_utils

from .conditional_decode import DirectTailDecode, normalize_for_cosine
from .ifelse_bridge import conditional_capture


@triton.jit(do_not_specialize=["threshold", "depth"])
def _write_predicates(scores, slots, selected, exits, continues, threshold, depth,
                      ROWS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.arange(0, BLOCK)
    score = tl.load(scores + row, row < ROWS, other=0).to(tl.float32)
    slot = tl.load(slots + row, row < ROWS, other=-1)
    failures = (slot >= 0) & ~(score > threshold)
    passes = tl.sum(failures.to(tl.int32), axis=0) == 0
    tl.store(selected, tl.where(passes, depth, 0).to(tl.int64))
    tl.store(exits, passes)
    tl.store(continues, ~passes)


def write_predicates(scores, slots, threshold, depth, selected, exits, continues):
    rows = scores.numel()
    if (scores.ndim != 1 or slots.shape != scores.shape or not 1 <= rows <= 64
            or scores.dtype != torch.bfloat16 or slots.dtype != torch.long
            or not scores.is_contiguous() or not slots.is_contiguous()
            or scores.device.type != "cuda" or slots.device != scores.device
            or selected.dtype != torch.long or exits.dtype != torch.bool or continues.dtype != torch.bool
            or any(t.shape != () or t.device != scores.device for t in (selected, exits, continues))):
        raise ValueError("Expected 1–64 BF16 scores, int64 slots and three CUDA scalar outputs")
    rounded = torch.tensor(threshold, dtype=torch.bfloat16, device="cpu").item()
    _write_predicates[(1,)](scores, slots, selected, exits, continues, rounded, depth,
                           ROWS=rows, BLOCK=triton.next_power_of_2(rows), num_warps=1)


class IfElseDecode(DirectTailDecode):
    def __init__(self, body, threshold, norm_weights, norm_eps, *, pool):
        super().__init__(body, threshold, norm_weights, norm_eps)
        self.pool = pool
        # Each active nested capture needs a distinct auxiliary stream.
        self.stages = [(torch.cuda.Stream(device=norm_weights.device), torch.cuda.Event(), torch.cuda.Event())
                       for _ in range(body.end_layer - 2)]
        flags = torch.empty((len(self.stages), 2), dtype=torch.bool, device=norm_weights.device)
        self.flags = [tuple(row.unbind()) for row in flags.unbind()]

    def exclude_capture_stream(self, root):
        # PyTorch's finite stream pool can give a stage the same stream as the root.
        used = {stage[0].cuda_stream for stage in self.stages} | {root.cuda_stream}
        for index, (auxiliary, ready, checked) in enumerate(self.stages):
            if auxiliary.cuda_stream == root.cuda_stream:
                replacement = torch.cuda.Stream(device=root.device)
                while replacement.cuda_stream in used:
                    replacement = torch.cuda.Stream(device=root.device)
                self.stages[index] = replacement, ready, checked
                break

    def warm(self, positions, ids):
        self.exclude_capture_stream(torch.cuda.current_stream(ids.device))
        super().warm(positions, ids)
        slots = next(iter(get_forward_context().slot_mapping.values()))
        exits, continues = self.flags[0]
        scores = torch.zeros_like(slots, dtype=torch.bfloat16)
        write_predicates(scores, slots, self.threshold, 2, slots.new_empty(()), exits, continues)

    def check_predicates(self, snapshot, before, effective_depth, selected, exits, continues):
        after = normalize_for_cosine(snapshot, self.epsilon)
        scores = (after * before).sum(dim=-1)
        slots = next(iter(get_forward_context().slot_mapping.values()))
        write_predicates(scores, slots, self.threshold, effective_depth, selected, exits, continues)
        return after

    def forward(self, positions, ids, output, selected_depth):
        body = self.body

        def finish(depth, hidden, residual):
            for layer in body.layers[depth:]:
                hidden, residual = layer._forward_quant(positions, hidden, residual)
            output.copy_(self.normalize(hidden, residual, selected_depth))

        def advance(depth, hidden, residual, before):
            next_depth = depth + 1
            layer = body.layers[depth]
            if next_depth == body.end_layer:
                hidden, residual = layer(positions, hidden, residual)
                selected_depth.fill_(next_depth)
                finish(next_depth, hidden, residual)
                return

            snapshot = hidden + residual
            auxiliary, ready, checked = self.stages[depth - 1]
            exits, continues = self.flags[depth - 1]
            main = torch.cuda.current_stream(ids.device)
            handle = dispatch.new_handle()
            ready.record(main)
            with torch.cuda.stream(auxiliary):
                auxiliary.wait_event(ready)
                after = self.check_predicates(snapshot, before, next_depth, selected_depth, exits, continues)
                dispatch.set_handle(handle, exits)
                checked.record(auxiliary)
            hidden, residual = layer(positions, hidden, residual)
            main.wait_event(checked)
            dispatch.capture_if_else(handle, lambda: finish(next_depth, hidden, residual),
                                     lambda: advance(next_depth, hidden, residual, after))

        # Only one precision branch executes. Reuse the Marlin workspace warmed on the root.
        root = torch.cuda.current_stream(ids.device)
        with patch.object(torch_utils, "current_stream", return_value=root), conditional_capture(self.pool) as dispatch:
            hidden = body.embed_input_ids(ids)
            before = normalize_for_cosine(hidden, self.epsilon)
            hidden, residual = body.layers[0](positions, hidden, None)
            advance(1, hidden, residual, before)
