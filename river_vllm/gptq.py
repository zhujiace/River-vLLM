"""River model registration with vLLM's native GPTQ tail."""

from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig
from vllm.model_executor.models.llama import LlamaForCausalLM

from . import model as native_river

ARCHITECTURE = "RiverGPTQLlamaForCausalLM"
WEIGHT_FORMAT = "gptq_w4_g128"
RIVER_OPTIONS = {
    "threshold": .5, "fixed_depth": None, "batch_mode": "legacy_all",
    "trace": False, "prefill_policy": "bf16",
}


class RiverGPTQModel(native_river.RiverModel):
    def __init__(self, *, vllm_config, prefix=""):
        config = vllm_config.model_config.hf_config
        if getattr(config, "river_weight_format", None) != WEIGHT_FORMAT:
            raise ValueError(f"Expected River weight format {WEIGHT_FORMAT!r}")
        quant = AutoGPTQConfig.from_config({
            "bits": 4, "group_size": 128, "desc_act": False,
            "sym": True, "lm_head": False, "quant_method": "gptq",
        })
        quant.packed_modules_mapping = RiverGPTQLlamaForCausalLM.packed_modules_mapping
        super().__init__(vllm_config=vllm_config, tail_quant_config=quant, prefix=prefix)

    def load_weights(self, weights):
        mapper = AutoGPTQConfig.get_checkpoint_weight_mapper()
        return super().load_weights(mapper.apply(weights))


class RiverGPTQLlamaForCausalLM(LlamaForCausalLM):
    def _init_model(self, vllm_config, prefix="", **kwargs):
        return RiverGPTQModel(vllm_config=vllm_config, prefix=prefix)
