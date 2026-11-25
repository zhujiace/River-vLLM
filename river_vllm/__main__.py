"""Offline River generation using the native vLLM decode graph."""

import argparse
import json
from pathlib import Path

from .runtime import DECODE_GRAPHS, build_llm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--decode-graph", choices=DECODE_GRAPHS, default="direct",
                        help="direct exit, or a one-layer delayed native IF/ELSE graph")
    parser.add_argument("--capture-sizes", type=int, nargs="+", default=[1])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--prompt", default="Explain what a key-value cache does in one sentence.")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= max(args.capture_sizes):
        parser.error("--batch-size must fit the largest capture size")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")

    llm = build_llm(args.model, tokenizer=args.tokenizer, threshold=args.threshold,
                    capture_sizes=args.capture_sizes, decode_graph=args.decode_graph)
    from vllm import SamplingParams

    params = SamplingParams(temperature=0, max_tokens=args.max_tokens)
    prompts = [args.prompt] * args.batch_size
    for output in llm.generate(prompts, params, use_tqdm=False):
        result = output.outputs[0]
        print(json.dumps({"request_id": output.request_id, "text": result.text,
                          "token_ids": list(result.token_ids)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
