"""CPU scheduler phases scoped to one native runner execution."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class BatchPhases:
    request_ids: tuple
    token_counts: tuple
    is_prefilling: tuple

    def flags_for(self, starts):
        expected = [0]
        for count in self.token_counts:
            expected.append(expected[-1] + int(count))
        actual = list(starts)
        while len(actual) > len(expected) and actual[-1] == actual[-2]:
            actual.pop()
        if (actual != expected or len(self.is_prefilling) != len(self.token_counts)
                or len(self.request_ids) != len(self.token_counts)):
            raise ValueError('River phase metadata does not match packed request boundaries')
        return list(self.is_prefilling)


_phases = ContextVar('river_batch_phases', default=None)


def current_phases():
    return _phases.get()


def set_phases(batch):
    _phases.set(batch)


@contextmanager
def phase_scope(batch=None):
    token = _phases.set(batch)
    try:
        yield
    finally:
        _phases.reset(token)
