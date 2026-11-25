"""Load the capture-only CUDA extension for a two-body conditional node."""

from contextlib import contextmanager
from functools import cache
from pathlib import Path


@cache
def load_extension():
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("The IF/ELSE decoder needs the CUDA development toolkit and CUDA_HOME")
    source = Path(__file__).with_name("ifelse_bridge.cpp")
    bridge = load("river_vllm_ifelse_cuda", sources=[str(source)], with_cuda=True, extra_cflags=["-O2"])
    bridge.check_allocator()
    return bridge


@contextmanager
def conditional_capture(pool):
    capture = load_extension().Capture(pool)
    try:
        yield capture
    finally:
        capture.close()
