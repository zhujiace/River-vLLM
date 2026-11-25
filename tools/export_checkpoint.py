"""Build the River GPTQ checkpoint from the measured BF16 and W4 sources."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from safetensors import safe_open
from safetensors.torch import save_file
import torch


GROUP = 128
FIELDS = ("qweight", "qzeros", "scales", "g_idx")


def map_projections(layers):
    projections = {
        ("self_attn", "quant_attn"): ("q_proj", "k_proj", "v_proj", "o_proj"),
        ("mlp", "quant_mlp"): ("gate_proj", "up_proj", "down_proj"),
    }
    return {
        f"model.layers.{layer}.{source}.{name}": f"model.layers.{layer}.{target}.{name}"
        for layer in layers
        for (source, target), names in projections.items()
        for name in names
    }


def validate_gptq(tensors, name):
    k = tensors["g_idx"].numel()
    n = tensors["qweight"].shape[1]
    if k % GROUP or n % 8:
        raise ValueError(f"Unsupported GPTQ projection dimensions for {name}: {(k, n)}")
    expected = {
        "qweight": ((k // 8, n), torch.int32),
        "qzeros": ((k // GROUP, n // 8), torch.int32),
        "scales": ((k // GROUP, n), torch.float16),
        "g_idx": ((k,), torch.int32),
    }
    for field, (shape, dtype) in expected.items():
        if tensors[field].shape != shape or tensors[field].dtype != dtype:
            raise ValueError(f"Unexpected {field} shape or dtype for {name}")
    if not torch.all(tensors["qzeros"] == 0x77777777):
        raise ValueError(f"Expected packed legacy center 8 in qzeros for {name}")
    if not torch.equal(tensors["g_idx"], torch.arange(k, dtype=torch.int32) // GROUP):
        raise ValueError(f"Expected contiguous g_idx for {name}")
    if not torch.all(torch.isfinite(tensors["scales"]) & (tensors["scales"] > 0)):
        raise ValueError(f"Invalid scales for {name}")


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_checkpoint(output, base, config, state, audit):
    with TemporaryDirectory(prefix=".river-gptq-", dir=output.parent) as staging:
        checkpoint = Path(staging) / "checkpoint"
        checkpoint.mkdir()
        weight_path = checkpoint / "model.safetensors"
        save_file(state, weight_path, metadata={"format": "pt"})
        (checkpoint / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                     "generation_config.json"):
            shutil.copyfile(base / name, checkpoint / name)
        audit["output_weight_sha256"] = file_hash(weight_path)
        (checkpoint / "export-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        checkpoint.rename(output)


def export(base, gptq, output):
    if output.exists():
        raise FileExistsError(output)
    config = json.loads((base / "config.json").read_text())
    if (config["architectures"] != ["LlamaForCausalLM"] or config["num_hidden_layers"] != 16
            or config["hidden_size"] != 2048 or not config["tie_word_embeddings"]):
        raise ValueError("Expected the Llama-3.2-1B-Instruct BF16 backbone")
    quantization = json.loads((gptq / "config.json").read_text())["quantization_config"]
    expected = dict(bits=4, group_size=GROUP, sym=True, desc_act=False, quant_method="gptq")
    if any(quantization.get(key) != value for key, value in expected.items()):
        raise ValueError("Expected symmetric GPTQ W4/G128 with desc_act disabled")

    depths = list(range(1, 16))
    projections = map_projections(depths)
    paths = {"base": base / "model.safetensors", "gptq": gptq / "model.safetensors"}
    with (safe_open(paths["base"], framework="pt") as bf16,
          safe_open(paths["gptq"], framework="pt") as w4):
        state = {name: bf16.get_tensor(name) for name in bf16.keys()}
        if any(value.dtype != torch.bfloat16 for value in state.values()):
            raise ValueError("Expected BF16 backbone tensors")
        for depth in depths:
            state[f"model.exit_norms.{depth - 1}.weight"] = state["model.norm.weight"].clone()
            prefix = f"model.layers.{depth}"
            for norm in ("input_layernorm", "post_attention_layernorm"):
                state[f"{prefix}.quant_{norm}.weight"] = state[f"{prefix}.{norm}.weight"].clone()
        for source, target in projections.items():
            tensors = {field: w4.get_tensor(f"{source}.{field}") for field in FIELDS}
            validate_gptq(tensors, source)
            if tuple(state[f"{source}.weight"].shape) != (
                    tensors["qweight"].shape[1], tensors["g_idx"].numel()):
                raise ValueError(f"GPTQ dimensions do not match {source}.weight")
            state.update({f"{target}.{field}": value for field, value in tensors.items()})

    config.update(
        architectures=["RiverGPTQLlamaForCausalLM"],
        exit_arch="river", exit_decoder_layer=True, exit_layer_indices=depths,
        output_exit_layers=depths, output_full_model=True, sdw_intermediate_size=4096,
        quant_bits=4, tie_exit_lm_head=True, exit_attention=False, exit_mlp=False,
        collect_kv=False, quantize_full_model=False, quantize_lm_head=False,
        transformers_version="4.50.0", river_weight_format="gptq_w4_g128",
        river_options=dict(threshold=.5, fixed_depth=None, prefill_policy="bf16",
                           batch_mode="legacy_all", trace=False),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    audit = dict(
        source={name: dict(sha256=file_hash(path)) for name, path in paths.items()},
        total_tensors=len(state), tied_head=True, gptq_configuration=expected,
        gptq_projection_map=projections, gptq_tensor_fields=FIELDS,
    )
    write_checkpoint(output, base, config, state, audit)
    print(f"Exported {len(state)} tensors to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--gptq", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.base, args.gptq, args.output)
