from __future__ import annotations
from typing import Iterable

from torch.nn import Module

from self_play_pretrain_zero_data.executors import Executor

# classes

class SelfPlay(Module):
    def __init__(
        self,
        generator: Module,
        learner: Module,
        executor: Executor
    ):
        super().__init__()
        self.generator = generator
        self.learner = learner
        self.executor = executor

    def execute(
        self,
        programs: str | Iterable[str],
        seed = None,
        **kwargs
    ) -> str | list[str]:

        if isinstance(programs, str):
            assert not isinstance(seed, Iterable), 'seed must be a single int or None for a single program'
            return self.executor(programs, seed = seed, **kwargs)

        if isinstance(seed, Iterable):
            return [self.executor(program, seed = s, **kwargs) for program, s in zip(programs, seed, strict = True)]

        return [
            self.executor(program, seed = (seed + i) if seed is not None else None, **kwargs)
            for i, program in enumerate(programs)
        ]

    def forward(self, *args, **kwargs):
        raise NotImplementedError
