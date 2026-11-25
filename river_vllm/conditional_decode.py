"""GPU-side River exit decisions and contiguous W4 branches."""

import torch
import triton
import triton.language as tl
from vllm import ir
from vllm.forward_context import get_forward_context


@triton.jit(do_not_specialize=["threshold", "depth"])
def _write_exit_depth(scores, slots, output, threshold, depth,
                      ROWS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.arange(0, BLOCK)
    score = tl.load(scores + row, row < ROWS, other=0).to(tl.float32)
    slot = tl.load(slots + row, row < ROWS, other=-1)
    failures = (slot >= 0) & ~(score > threshold)
    passes = tl.sum(failures.to(tl.int32), axis=0) == 0
    tl.store(output, tl.where(passes, depth, 0).to(tl.int64))


def write_exit_depth(scores, slots, threshold, depth, output):
    rows = scores.numel()
    if (scores.ndim != 1 or slots.shape != scores.shape or not 1 <= rows <= 64
            or scores.dtype != torch.bfloat16 or slots.dtype != torch.long
            or output.shape != () or output.dtype != torch.long
            or scores.device.type != "cuda" or slots.device != scores.device
            or output.device != scores.device or not scores.is_contiguous()
            or not slots.is_contiguous()):
        raise ValueError("Expected up to 64 contiguous BF16 scores and int64 slots on CUDA")
    # Match the scalar conversion of PyTorch's BF16 score comparison.
    rounded = torch.tensor(threshold, dtype=torch.bfloat16, device="cpu").item()
    _write_exit_depth[(1,)](scores, slots, output, rounded, depth,
                           ROWS=rows, BLOCK=triton.next_power_of_2(rows), num_warps=1)


def run_if(predicate, operation):
    graph = torch.cuda.CUDAGraph.get_currently_capturing_graph()
    graph.begin_capture_to_if_node(predicate)
    try:
        operation()
    finally:
        graph.end_capture_to_conditional_node()


def normalize_for_cosine(hidden, epsilon):
    norm = torch.linalg.vector_norm(hidden, ord=2, dim=-1, keepdim=True)
    norm.clamp_min_(epsilon)
    return hidden / norm


class DirectTailDecode:
    def __init__(self, body, threshold, norm_weights, norm_eps):
        self.body = body
        self.threshold = threshold
        self.norm_weights = norm_weights
        self.norm_eps = norm_eps
        self.epsilon = torch.scalar_tensor(1e-8, dtype=torch.float32, device=norm_weights.device)

    def warm(self, positions, ids):
        # Initialize BF16 and Marlin kernels and warm the exit reduction.
        for depth in (1, self.body.end_layer):
            self.body._fixed_forward(positions, self.body.embed_input_ids(ids), depth)
        slots = next(iter(get_forward_context().slot_mapping.values()))
        scores = torch.zeros_like(slots, dtype=torch.bfloat16)
        write_exit_depth(scores, slots, self.threshold, 1, slots.new_empty(()))

    def normalize(self, hidden, residual, depth):
        weight = torch.index_select(self.norm_weights, 0, depth.reshape(1) - 1)[0]
        return ir.ops.fused_add_rms_norm.maybe_inplace(hidden, residual, weight, self.norm_eps)[0]

    def forward(self, positions, ids, output, selected_depth):
        body = self.body

        def finish(depth, hidden, residual):
            for layer in body.layers[depth:]:
                hidden, residual = layer._forward_quant(positions, hidden, residual)
            output.copy_(self.normalize(hidden, residual, selected_depth))

        def bf16_prefix(depth, hidden, residual, before):
            layer = body.layers[depth - 1]
            if depth == body.end_layer:
                hidden, residual = layer(positions, hidden, residual)
                selected_depth.fill_(depth)
                finish(depth, hidden, residual)
                return

            hidden, residual = layer(positions, hidden, residual)
            after = normalize_for_cosine(hidden + residual, self.epsilon)
            scores = (after * before).sum(dim=-1)
            slots = next(iter(get_forward_context().slot_mapping.values()))
            write_exit_depth(scores, slots, self.threshold, depth, selected_depth)
            exits = selected_depth.ne(0)
            # Both siblings use the saved predicate. The continuation can update depth.
            run_if(exits, lambda: finish(depth, hidden, residual))
            run_if(~exits, lambda: bf16_prefix(depth + 1, hidden, residual, after))

        hidden = body.embed_input_ids(ids)
        bf16_prefix(1, hidden, None, normalize_for_cosine(hidden, self.epsilon))
