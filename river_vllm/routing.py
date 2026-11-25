"""Plain eager routing, retaining sequence-prefill and token-decode semantics."""
import torch
from torch.nn import functional as F


def apply_by_rows(bf16, quant, inputs, low_mask):
    """Execute only populated branches, restoring every output to its input row."""
    if bool(low_mask.all()):return quant(*inputs)
    if not bool(low_mask.any()):return bf16(*inputs)
    result = None
    tuple_result = False
    for mask, operation in [(~low_mask, bf16), (low_mask, quant)]:
        indices = torch.nonzero(mask, as_tuple=False).flatten()
        selected = tuple(value.index_select(0, indices) if value is not None else None
                         for value in inputs)
        output = operation(*selected)
        tuple_result = isinstance(output, tuple)
        output = output if tuple_result else (output,)
        if result is None:
            result = [torch.empty((len(low_mask), *value.shape[1:]),
                                  device=value.device, dtype=value.dtype) for value in output]
        for target, value in zip(result, output):target.index_copy_(0, indices, value)
    return tuple(result) if tuple_result else result[0]


class RouteController:
    def __init__(self, query_starts, num_layers, exit_layers, threshold,
                 fixed_depth=None, batch_mode='independent', *,
                 is_prefilling=None, prefill_policy='adaptive'):
        starts = list(query_starts)
        if not starts or starts[0] != 0 or len(starts) < 2:
            raise ValueError('Query boundaries must begin at zero')
        while len(starts) > 2 and starts[-1] == starts[-2]:
            starts.pop()
        if any(a >= b for a, b in zip(starts, starts[1:])):
            raise ValueError('Only trailing empty request slots are supported')
        if batch_mode not in {'independent', 'legacy_all'}:
            raise ValueError('Unknown batch routing rule')
        if fixed_depth is not None and fixed_depth not in {*exit_layers, num_layers}:
            raise ValueError('Fixed depth has no corresponding final norm')
        self.starts = starts
        self.num_requests = len(starts)-1
        if prefill_policy not in {'bf16', 'adaptive'}:
            raise ValueError('Unknown River prefill policy')
        if prefill_policy == 'bf16' and (is_prefilling is None
                or len(is_prefilling) != self.num_requests):
            raise ValueError('Full BF16 prefill requires scheduler request phases')
        self.force_full = ([bool(flag) for flag in is_prefilling]
                           if prefill_policy == 'bf16' else [False]*self.num_requests)
        self.actual_tokens = starts[-1]
        self.num_layers = num_layers
        self.exit_layers = tuple(exit_layers)
        self.threshold = threshold
        self.fixed_depth = fixed_depth
        self.batch_mode = batch_mode
        self.exit_depths = [num_layers if fixed_depth is None or full else fixed_depth
                            for full in self.force_full]
        self.exited = [False]*self.num_requests
        self.precision_mode = 'bf16'
        self.checks = 0

    def observe(self, depth, before, after):
        if all(done or full for done, full in zip(self.exited, self.force_full)):
            return
        if self.fixed_depth is not None:
            if depth == self.fixed_depth:
                self._record(depth, [not full for full in self.force_full])
            return
        if self.threshold is None or depth not in self.exit_layers:
            return
        scores = F.cosine_similarity(after[:self.actual_tokens],
                                     before[:self.actual_tokens], dim=-1)
        candidates = torch.stack([(scores[a:b] > self.threshold).all()
                                  for a, b in zip(self.starts, self.starts[1:])])
        if self.batch_mode == 'legacy_all':
            eligible = torch.tensor([not full for full in self.force_full],
                                    device=candidates.device, dtype=torch.bool)
            candidates = candidates[eligible].all().expand(self.num_requests)
        # This host decision is intentional in the unoptimized reference.
        decisions = candidates.detach().cpu().tolist()
        self.checks += 1
        self._record(depth, decisions)

    def _record(self, depth, decisions):
        for i, decision in enumerate(decisions):
            if decision and not self.exited[i] and not self.force_full[i]:
                self.exited[i] = True
                self.exit_depths[i] = depth
        self.precision_mode = ('quant' if all(self.exited) else
                               'mixed' if any(self.exited) else 'bf16')

    def use_quant_at(self, layer_index):
        return self.fixed_depth is not None and layer_index >= self.fixed_depth

    def low_precision_mask(self, num_rows, device):
        if num_rows < self.actual_tokens:
            raise ValueError('Packed rows are smaller than the real token count')
        flags = [False]*num_rows
        for active, start, end in zip(self.exited, self.starts, self.starts[1:]):
            if active:flags[start:end] = [True]*(end-start)
        return torch.tensor(flags, device=device, dtype=torch.bool)

    def norm_groups(self, num_rows):
        groups = {}
        for depth, start, end in zip(self.exit_depths, self.starts, self.starts[1:]):
            groups.setdefault(depth, []).extend(range(start, end))
        if num_rows > self.actual_tokens:
            groups.setdefault(self.num_layers, []).extend(range(self.actual_tokens, num_rows))
        return groups
