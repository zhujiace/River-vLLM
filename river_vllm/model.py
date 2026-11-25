# SPDX-License-Identifier: Apache-2.0
# Reuses the vLLM Llama implementation, Copyright contributors to the vLLM project.
"""Minimal TP1 River adapter: two precisions, one native Attention per layer."""
import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.llama import LlamaDecoderLayer, LlamaMLP, LlamaModel
from vllm.model_executor.models.utils import (
    AutoWeightsLoader, extract_layer_index, make_empty_intermediate_tensors_factory,
)

from .routing import RouteController, apply_by_rows
from .phases import current_phases


def snapshot_hidden(hidden_states, residual):
    # The first residual aliases the embedding; native fused norm updates it.
    return hidden_states.clone() if residual is None else hidden_states + residual


class RiverDecoderLayer(LlamaDecoderLayer):
    def __init__(self, vllm_config, tail_quant_config, prefix=''):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        index = extract_layer_index(prefix)
        if index == 0:
            return
        config = vllm_config.model_config.hf_config
        attention = self.self_attn
        # Projections only: creating a second LlamaAttention would allocate a
        # second logical KV bank and give each branch incomplete history.
        self.quant_attn = nn.Module()
        self.quant_attn.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size, head_size=attention.head_dim,
            total_num_heads=config.num_attention_heads,
            total_num_kv_heads=config.num_key_value_heads, bias=False,
            quant_config=tail_quant_config, prefix=f'{prefix}.quant_attn.qkv_proj')
        self.quant_attn.o_proj = RowParallelLinear(
            input_size=attention.q_size, output_size=config.hidden_size,
            bias=False, quant_config=tail_quant_config, prefix=f'{prefix}.quant_attn.o_proj')
        self.quant_mlp = LlamaMLP(
            hidden_size=config.hidden_size, intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act, quant_config=tail_quant_config,
            prefix=f'{prefix}.quant_mlp')
        self.quant_input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.quant_post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def _forward_quant(self, positions, hidden_states, residual):
        if residual is None:
            residual = hidden_states
            hidden_states = self.quant_input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.quant_input_layernorm(hidden_states, residual)
        qkv = self.quant_attn.qkv_proj(hidden_states)[0]
        attention = self.self_attn
        q, k, v = qkv.split([attention.q_size, attention.kv_size, attention.kv_size], -1)
        q, k = attention.rotary_emb(positions, q, k)
        hidden_states = self.quant_attn.o_proj(attention.attn(q, k, v))[0]
        hidden_states, residual = self.quant_post_attention_layernorm(hidden_states, residual)
        return self.quant_mlp(hidden_states), residual

    def _forward_mixed(self, positions, hidden_states, residual, low_mask):
        hidden_states, residual = apply_by_rows(
            self.input_layernorm, self.quant_input_layernorm,
            (hidden_states, residual), low_mask)
        qkv = apply_by_rows(
            lambda x: self.self_attn.qkv_proj(x)[0],
            lambda x: self.quant_attn.qkv_proj(x)[0], (hidden_states,), low_mask)
        attention = self.self_attn
        q, k, v = qkv.split([attention.q_size, attention.kv_size, attention.kv_size], -1)
        q, k = attention.rotary_emb(positions, q, k)
        # Original packed row order is mandatory for native slot mappings.
        hidden_states = attention.attn(q, k, v)
        hidden_states = apply_by_rows(
            lambda x: self.self_attn.o_proj(x)[0],
            lambda x: self.quant_attn.o_proj(x)[0], (hidden_states,), low_mask)
        hidden_states, residual = apply_by_rows(
            self.post_attention_layernorm, self.quant_post_attention_layernorm,
            (hidden_states, residual), low_mask)
        hidden_states = apply_by_rows(self.mlp, self.quant_mlp, (hidden_states,), low_mask)
        return hidden_states, residual


@support_torch_compile(dynamic_arg_dims={
    'input_ids': {0: 'b'}, 'positions': {0: 'b'},
    'intermediate_tensors': {0: 'b'}, 'inputs_embeds': {0: 'b'},
})
class RiverModel(nn.Module):
    hf_to_vllm_mapper = LlamaModel.hf_to_vllm_mapper

    def __init__(self, *, vllm_config, tail_quant_config, prefix='', **kwargs):
        super().__init__()
        parallel = vllm_config.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError('Initial River adapter supports TP1/PP1 only')
        if vllm_config.quant_config is not None:
            raise ValueError('River uses explicit tail quantization, not global quantization')
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError('River initial validation requires prefix caching disabled')
        if vllm_config.scheduler_config.enable_chunked_prefill:
            raise ValueError('River sequence-level exits require unchunked prefill')
        self.config = config = vllm_config.model_config.hf_config
        if config.num_hidden_layers != 16 or config.exit_layer_indices != list(range(1, 16)):
            raise ValueError('Initial River adapter targets the verified contiguous 1B checkpoint')
        self.options = dict(config.river_options)
        self.options.setdefault('prefill_policy', 'bf16')
        if self.options['prefill_policy'] not in {'bf16', 'adaptive'}:
            raise ValueError('Unknown River prefill policy')
        self.fixed_depth = self.options.get('fixed_depth')
        self.trace = self.options.get('trace', False)
        self.route_history = []
        self.last_slots = {}
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            RiverDecoderLayer(vllm_config, tail_quant_config, f'{prefix}.layers.{i}')
            for i in range(config.num_hidden_layers)])
        self.start_layer = 0
        self.end_layer = config.num_hidden_layers
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.exit_norms = nn.ModuleList([
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            for _ in config.exit_layer_indices])
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ['hidden_states', 'residual'], config.hidden_size)
        # Validate fixed depth even before native profiling/warmup.
        RouteController([0, 1], config.num_hidden_layers, config.exit_layer_indices,
                        self.options.get('threshold', .5), self.fixed_depth,
                        self.options.get('batch_mode', 'independent'))

    def embed_input_ids(self, input_ids):
        return self.embed_tokens(input_ids)

    def _norm_for(self, depth):
        return self.norm if depth == self.end_layer else self.exit_norms[depth-1]

    def _fixed_forward(self, positions, hidden_states, depth):
        residual = None
        for index, layer in enumerate(self.layers):
            if index < depth:
                hidden_states, residual = layer(positions, hidden_states, residual)
            else:
                hidden_states, residual = layer._forward_quant(positions, hidden_states, residual)
        return self._norm_for(depth)(hidden_states, residual)[0]

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        hidden_states = (self.embed_input_ids(input_ids) if inputs_embeds is None else inputs_embeds)
        if self.fixed_depth is not None and (self.options['prefill_policy'] == 'adaptive'
                                             or self.fixed_depth == self.end_layer):
            return self._fixed_forward(positions, hidden_states, self.fixed_depth)
        context = get_forward_context()
        metadata = context.attn_metadata
        if metadata is None:
            if torch.compiler.is_compiling():
                raise RuntimeError('River request boundaries are absent during vLLM graph tracing; use eager')
            # Native dummy profiling has no requests or KV metadata. Allocate
            # the full BF16 path, which bounds this minimal adapter's workload.
            return self._fixed_forward(positions, hidden_states, self.end_layer)
        if not isinstance(metadata, dict):
            raise ValueError('River initial adapter does not support microbatch metadata')
        first = next(iter(metadata.values()))
        starts = first.query_start_loc.detach().cpu().tolist()
        phases = current_phases()
        if self.options['prefill_policy'] == 'bf16' and phases is None:
            raise ValueError('Full BF16 prefill needs worker_cls=river_vllm.worker.RiverWorker')
        flags = phases.flags_for(starts) if phases is not None else None
        route = RouteController(
            starts, self.end_layer, self.config.exit_layer_indices,
            self.options.get('threshold', .5), fixed_depth=self.fixed_depth,
            batch_mode=self.options.get('batch_mode', 'independent'),
            is_prefilling=flags, prefill_policy=self.options['prefill_policy'])
        residual = None
        mixed_layers = 0
        for index, layer in enumerate(self.layers):
            before = snapshot_hidden(hidden_states, residual)
            if route.precision_mode == 'bf16':
                hidden_states, residual = layer(positions, hidden_states, residual)
            elif route.precision_mode == 'quant':
                hidden_states, residual = layer._forward_quant(positions, hidden_states, residual)
            else:
                mixed_layers += 1
                mask = route.low_precision_mask(hidden_states.shape[0], hidden_states.device)
                hidden_states, residual = layer._forward_mixed(positions, hidden_states, residual, mask)
            route.observe(index+1, before, hidden_states + residual)
        groups = route.norm_groups(hidden_states.shape[0])
        if len(groups) == 1:
            output = self._norm_for(next(iter(groups)))(hidden_states, residual)[0]
        else:
            output = torch.empty_like(hidden_states)
            for depth, rows in groups.items():
                indices = torch.tensor(rows, device=hidden_states.device)
                values = self._norm_for(depth)(hidden_states.index_select(0, indices),
                                               residual.index_select(0, indices))[0]
                output.index_copy_(0, indices, values)
        if self.trace:
            self.route_history.append({
                'starts': route.starts, 'positions': positions[:route.actual_tokens].cpu().tolist(),
                'exit_depths': route.exit_depths, 'checks': route.checks, 'mixed_layers': mixed_layers,
                'is_prefilling': flags, 'prefill_policy': self.options['prefill_policy']})
            # Diagnostics only. Never enable this during ordinary timings.
            self.last_slots = {name: meta.slot_mapping[:route.actual_tokens].detach().clone()
                               for name, meta in metadata.items()}
        return output

    def load_weights(self, weights):
        return AutoWeightsLoader(self).load_weights(weights, mapper=self.hf_to_vllm_mapper)
