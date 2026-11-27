"""Construct the single-GPU V2 runner used by the River graph release."""

import os


DEFAULT_DECODE_GRAPH = "latched"
DECODE_GRAPHS = (DEFAULT_DECODE_GRAPH, "direct", "ifelse")


def build_llm(model, *, tokenizer=None, threshold=0.5, capture_sizes=(1,), decode_graph=DEFAULT_DECODE_GRAPH):
    """Load a local River GPTQ checkpoint and capture its decode shapes."""
    sizes = sorted(set(capture_sizes))
    if not sizes or sizes[0] < 1 or sizes[-1] > 64:
        raise ValueError("Capture sizes must be between 1 and 64")
    if not 0 <= threshold <= 1:
        raise ValueError("The exit threshold must be between 0 and 1")
    if decode_graph not in DECODE_GRAPHS:
        raise ValueError(f"decode_graph must be one of {DECODE_GRAPHS}")
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from vllm import LLM

    from . import register
    from .eager import install_eager_dispatch
    from .gptq import RIVER_OPTIONS

    register()
    llm = LLM(
        model=str(model), tokenizer=str(tokenizer or model), dtype="bfloat16", seed=42,
        max_model_len=2048, max_num_seqs=sizes[-1],
        max_num_batched_tokens=53248 if sizes[-1] > 32 else 32768,
        enforce_eager=False, enable_prefix_caching=False, enable_chunked_prefill=False,
        async_scheduling=False, gpu_memory_utilization=0.4, kv_cache_memory_bytes=2**31,
        generation_config="vllm", disable_log_stats=False,
        worker_cls="river_vllm.native.NativeGraphRiverWorker",
        attention_config={"backend": "FLASH_ATTN", "flash_attn_version": 2},
        kernel_config={"linear_backend": "marlin", "enable_jit_warmup": False},
        hf_overrides={
            "river_options": dict(RIVER_OPTIONS, threshold=threshold),
            "river_native_fixed": False,
            "river_decode_graph": decode_graph,
        },
        compilation_config={
            "mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": sizes,
        },
    )
    llm.collective_rpc(install_eager_dispatch)
    return llm
