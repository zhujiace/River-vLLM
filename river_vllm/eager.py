"""Phase-aware fallback for prefill and uncaptured steps."""

from itertools import accumulate
from unittest.mock import patch

import torch
from torch.nn import functional as F

from .model import snapshot_hidden
from .phases import current_phases


def by_indices(bf16, w4, inputs, row_sets):
    output = None
    for operation, indices in zip((bf16, w4), row_sets, strict=True):
        values = operation(*(value.index_select(0, indices) for value in inputs))
        single = isinstance(values, torch.Tensor)
        if single:
            values = (values,)
        if output is None:
            output = [v.new_empty((len(inputs[0]), *v.shape[1:])) for v in values]
        for target, value in zip(output, values, strict=True):
            target.index_copy_(0, indices, value)
    return output[0] if single else tuple(output)


def mixed_layer(layer, positions, hidden, residual, row_sets):
    attention = layer.self_attn
    hidden, residual = by_indices(
        layer.input_layernorm, layer.quant_input_layernorm, (hidden, residual), row_sets)
    qkv = by_indices(
        lambda x: attention.qkv_proj(x)[0], lambda x: layer.quant_attn.qkv_proj(x)[0],
        (hidden,), row_sets)
    q, k, v = qkv.split([attention.q_size, attention.kv_size, attention.kv_size], -1)
    q, k = attention.rotary_emb(positions, q, k)
    attention_output = attention.attn(q, k, v)
    hidden = by_indices(
        lambda x: attention.o_proj(x)[0], lambda x: layer.quant_attn.o_proj(x)[0],
        (attention_output,), row_sets)
    hidden, residual = by_indices(
        layer.post_attention_layernorm, layer.quant_post_attention_layernorm,
        (hidden, residual), row_sets)
    return by_indices(layer.mlp, layer.quant_mlp, (hidden,), row_sets), residual


class EagerDispatch:
    def __init__(self, body):
        self.body = body
        self.original = body.forward

    def pack(self, ids, positions, starts, ends, prefill):
        body = self.body
        hidden = body.embed_input_ids(ids)
        residual = None
        full = [row for a, b, flag in zip(starts, ends, prefill, strict=True) if flag for row in range(a, b)]
        decode = [a for a, flag in zip(starts, prefill, strict=True) if not flag]
        row_sets = tuple(torch.tensor(rows, device=ids.device, dtype=torch.long)
                         for rows in (full, decode))
        depth = body.end_layer
        for index, layer in enumerate(body.layers, 1):
            if index > depth:
                hidden, residual = mixed_layer(layer, positions, hidden, residual, row_sets)
                continue
            before = snapshot_hidden(hidden, residual) if index < depth else None
            hidden, residual = layer(positions, hidden, residual)
            if before is not None:
                scores = F.cosine_similarity(hidden + residual, before, dim=-1)
                if scores.index_select(0, row_sets[1]).gt(body.options["threshold"]).all().item():
                    depth = index
        if depth == body.end_layer:
            return body.norm(hidden, residual)[0], depth
        output, _ = by_indices(body.norm, body._norm_for(depth), (hidden, residual), row_sets)
        return output, depth

    def decode(self, positions, hidden):
        body = self.body
        residual = None
        depth = body.end_layer
        for index, layer in enumerate(body.layers, 1):
            before = snapshot_hidden(hidden, residual) if index < depth else None
            hidden, residual = layer(positions, hidden, residual)
            if (before is not None
                    and F.cosine_similarity(hidden + residual, before, dim=-1)
                    .gt(body.options["threshold"]).all().item()):
                depth = index
                break
        for layer in body.layers[depth:]:
            hidden, residual = layer._forward_quant(positions, hidden, residual)
        return body._norm_for(depth)(hidden, residual)[0]

    def __call__(self, ids, positions, intermediate_tensors=None, inputs_embeds=None):
        phases = current_phases()
        args = (ids, positions, intermediate_tensors, inputs_embeds)
        if phases is None or not phases.is_prefilling:
            return self.original(*args)
        prefill = phases.is_prefilling
        counts = phases.token_counts
        if all(prefill) and self.body.options["prefill_policy"] == "bf16":
            with patch.object(self.body, "fixed_depth", self.body.end_layer):
                return self.original(*args)
        if not any(prefill) and len(counts) == positions.numel() and counts == (1,) * len(counts):
            hidden = self.body.embed_input_ids(ids) if inputs_embeds is None else inputs_embeds
            return self.decode(positions, hidden)
        ends = tuple(accumulate(counts))
        if (any(prefill) and not all(prefill)
                and inputs_embeds is None and intermediate_tensors is None
                and ends[-1] == ids.numel()
                and all(flag or count == 1 for count, flag in zip(counts, prefill, strict=True))):
            output, _ = self.pack(ids, positions, (0, *ends[:-1]), ends, prefill)
            return output
        return self.original(*args)


def install_eager_dispatch(worker):
    body = worker.model_runner.get_model().model
    body.forward = EagerDispatch(body)
