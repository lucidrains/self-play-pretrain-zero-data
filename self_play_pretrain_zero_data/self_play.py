from __future__ import annotations
from typing import Iterable
from abc import ABC, abstractmethod

from torch.nn import Module

# executor

class ProgramExecutor(ABC):
    """maps a program string to its output string"""

    @abstractmethod
    def __call__(self, program: str, **kwargs) -> str:
        raise NotImplementedError

# main class

class SelfPlay(Module):
    def __init__(
        self,
        generator: Module,
        learner: Module,
        executor: ProgramExecutor
    ):
        super().__init__()
        self.generator = generator
        self.learner = learner
        self.executor = executor

    def execute(
        self,
        programs: str | Iterable[str],
        **kwargs
    ) -> str | list[str]:

        if isinstance(programs, str):
            return self.executor(programs, **kwargs)

        return [self.executor(program, **kwargs) for program in programs]

    def forward(self, *args, **kwargs):
        raise NotImplementedError
