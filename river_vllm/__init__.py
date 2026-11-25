"""Opt-in registration of the River GPTQ model with vLLM."""


def register():
    from vllm import ModelRegistry

    from .gptq import ARCHITECTURE, RiverGPTQLlamaForCausalLM

    ModelRegistry.register_model(ARCHITECTURE, RiverGPTQLlamaForCausalLM)
