"""Opt-in worker extension; vendor runner and ordinary model workers stay intact."""
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from .phases import BatchPhases, phase_scope, set_phases


class RiverModelRunner(GPUModelRunner):
    def gather_batch_req_state(self, scheduler_output, dummy_run):
        state, uniform_tokens = super().gather_batch_req_state(scheduler_output, dummy_run)
        set_phases(None if state is None else BatchPhases(
            tuple(state.req_ids), tuple(state.num_scheduled_tokens.tolist()),
            tuple(state.is_prefilling_np.tolist())))
        return state, uniform_tokens

    def execute_model(self, *args, **kwargs):
        with phase_scope():
            return super().execute_model(*args, **kwargs)
