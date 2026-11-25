"""Capture the River decoder with vLLM V2's own buffers and graph manager."""

from unittest.mock import patch

import torch
from vllm.config import CompilationMode, CUDAGraphMode
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.quantization.utils.marlin_utils import get_marlin_workspace

from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.worker.gpu_worker import Worker

from .conditional_decode import DirectTailDecode
from .worker import RiverModelRunner


def shared_attention_metadata(context, rows):
    metadata = context.attn_metadata
    if not isinstance(metadata, dict) or not metadata:
        return None
    shared = next(iter(metadata.values()))
    if not isinstance(shared, FlashAttentionMetadata) or any(m is not shared for m in metadata.values()):
        return None
    if (shared.num_actual_tokens != rows or shared.max_query_len != 1
            or shared.seq_lens.shape != (rows,) or shared.slot_mapping.shape != (rows,)
            or shared.query_start_loc.shape != (rows + 1,) or shared.block_table.shape[0] != rows
            or shared.scheduler_metadata is not None or shared.dcp_context_kv_lens is not None
            or shared.mm_prefix_query_range_tensor is not None or shared.rswa_prefix_lens is not None
            or shared.causal is not True or shared.max_num_splits != 0):
        return None
    slots = context.slot_mapping
    if not isinstance(slots, dict) or any(
            name not in slots or slots[name].data_ptr() != shared.slot_mapping.data_ptr()
            for name in metadata):
        return None
    return shared


class NativeGraphRiverRunner(RiverModelRunner):
    def gather_batch_req_state(self, scheduler_output, dummy_run):
        state, uniform_tokens = super().gather_batch_req_state(scheduler_output, dummy_run)
        # V2 can otherwise send one-token prefill extensions through a decode graph.
        if state is not None and state.has_prefill:
            uniform_tokens = None
        return state, uniform_tokens

    @torch.inference_mode()
    def capture_model(self, *, profile_only=False):
        if (self.compilation_config.mode != CompilationMode.NONE
                or self.cudagraph_manager.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY
                or self.parallel_config.world_size != 1 or self.dp_size != 1
                or self.decode_query_len != 1 or self.speculator is not None
                or self.supports_mm_inputs or self.lora_config is not None):
            raise ValueError("River native replay requires an uncompiled, single-GPU V2 decode graph")

        if not profile_only:
            # Capture warms its own stream. Mixed eager steps use the default stream
            # and need their Marlin locks before the V2 runner locks the workspace.
            with torch.cuda.stream(torch.cuda.default_stream(self.device)):
                get_marlin_workspace(self.device)

        body = self.get_model().model
        fixed = self.model_config.hf_config.river_native_fixed
        decoder = None
        if not fixed:
            norms = [body._norm_for(depth) for depth in range(1, body.end_layer + 1)]
            norm_eps = norms[-1].variance_epsilon
            if not all(norm.variance_epsilon == norm_eps and norm.pass_weight_add
                       and norm.variance_size_override is None for norm in norms):
                raise ValueError("River's decode norms must share RMSNorm settings")
            weights = torch.stack([norm.weight.detach() for norm in norms])
            decoder = DirectTailDecode(body, body.options["threshold"], weights, norm_eps)

        captured_depths = {}

        def forward(input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
            rows = positions.numel()
            source = shared_attention_metadata(get_forward_context(), rows)
            if (intermediate_tensors is not None or inputs_embeds is not None or source is None
                    or source.max_seq_len != self.max_model_len):
                raise ValueError("River native capture requires the shared FA2 decode metadata")
            if decoder is None or not torch.cuda.is_current_stream_capturing():
                if decoder is not None:
                    decoder.warm(positions, input_ids)
                return body._fixed_forward(positions, body.embed_input_ids(input_ids), body.end_layer)

            output = body.embed_tokens.weight.new_empty((rows, body.config.hidden_size))
            depth = input_ids.new_empty((), dtype=torch.long)
            decoder.forward(positions, input_ids, output, depth)
            captured_depths[rows] = depth
            return output

        # The original model forward handles prefill and ungraphed batches.
        # During actual decode the V2 manager replays the captured graph directly.
        with patch.object(body, "forward", forward):
            memory_bytes = super().capture_model(profile_only=profile_only)
        if not profile_only:
            self.river_decoder = decoder
            self.river_graph_depths = captured_depths
            self.river_graph_memory_bytes = memory_bytes
        return memory_bytes


class NativeGraphRiverWorker(Worker):
    def _make_model_runner(self):
        if not self.use_v2_model_runner:
            raise ValueError("River native replay requires Model Runner V2")
        return NativeGraphRiverRunner(self.vllm_config, self.device)
