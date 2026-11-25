"""Check native B1/B7 replay and a mixed prefill/decode step on one GPU."""

import argparse
from pathlib import Path

from river_vllm.runtime import DECODE_GRAPHS, build_llm


def install_probe(worker):
    runner = worker.model_runner
    manager = runner.cudagraph_manager
    dispatch = runner.get_model().model.forward
    run_fullgraph = manager.run_fullgraph
    pack = dispatch.pack
    worker.release_probe = []

    def replay(descriptor):
        worker.release_probe.append(("graph", descriptor.num_tokens))
        return run_fullgraph(descriptor)

    def mixed(*args):
        worker.release_probe.append(("mixed", len(args[-1])))
        return pack(*args)

    manager.run_fullgraph = replay
    dispatch.pack = mixed
    return sorted(descriptor.num_tokens for descriptor in manager.graphs)


def drain_probe(worker):
    events = worker.release_probe.copy()
    worker.release_probe.clear()
    return events


def check(model, decode_graph="direct", threshold=0.5):
    llm = build_llm(model, threshold=threshold, capture_sizes=(1, 2, 4, 8), decode_graph=decode_graph)
    from vllm import SamplingParams

    assert llm.collective_rpc(install_probe) == [[1, 2, 4, 8]]
    prompt = "A key-value cache is used in a language model to"
    params = SamplingParams(temperature=0, ignore_eos=True, max_tokens=8)
    for batch, capacity in ((1, 1), (7, 8)):
        result = llm.generate([prompt] * batch, params, use_tqdm=False)
        assert [len(item.outputs[0].token_ids) for item in result] == [8] * batch
        events, = llm.collective_rpc(drain_probe)
        assert events == [("graph", capacity)] * 7, events
        print(f"B{batch} replayed B{capacity} seven times.")

    # The first request finishes immediately, letting request 9 prefill alongside decode.
    short = SamplingParams(temperature=0, ignore_eos=True, max_tokens=1)
    result = llm.generate([prompt] * 9, [short] + [params] * 8, use_tqdm=False)
    assert [len(item.outputs[0].token_ids) for item in result] == [1] + [8] * 8
    events, = llm.collective_rpc(drain_probe)
    assert any(kind == "mixed" for kind, _ in events), events
    print("Nine queued requests exercised mixed-step indexing with expected output lengths.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--decode-graph", choices=DECODE_GRAPHS, default="direct")
    parser.add_argument("--threshold", type=float, default=0.5)
    options = parser.parse_args()
    check(options.model, options.decode_graph, options.threshold)
