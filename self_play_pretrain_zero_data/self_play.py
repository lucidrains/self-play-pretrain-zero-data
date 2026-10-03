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

    def execute_program(
        self,
        programs: str | Iterable[str],
        **kwargs
    ) -> str | list[str]:

        if isinstance(programs, str):
            return self.executor(programs, **kwargs)

        return [self.executor(program, **kwargs) for program in programs]

    def forward(self, *args, **kwargs):
        raise NotImplementedError
