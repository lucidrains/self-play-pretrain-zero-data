from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import chain, repeat

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# functions

def input_stream(
    input = '',
    cell_modulus = 256,
    seed = None,
    random_input = True
):
    """lazy cells - the input bytes first, then uniform random cells (or zeros)"""
    mod = max(cell_modulus, 1)
    bytes_in = (ord(c) for c in input) if isinstance(input, str) else input
    rng = random.Random(seed)
    fallback = (rng.randrange(mod) for _ in repeat(None)) if random_input else repeat(0)

    return chain((b % mod for b in bytes_in), fallback)

# classes

@dataclass(frozen = True)
class ExecutionInfo:
    """output and dynamics of one program run - the input to behavior descriptors"""

    program: str
    output: str = ''
    steps: int = 0  # instructions or words executed
    loops: int = 0  # backward control flow jumps, i.e. loop iterations

class Executor(ABC):
    """maps a program string to its output string

    token id 0 is reserved for sos / eos
    """

    num_tokens: int
    sos_eos_id = 0
    pad_id = -1

    @property
    def halt_id(self):
        """single-token program terminator, e.g. `F` for brainfuck, if defined"""

        halt_symbol = getattr(self, 'halt_symbol', None)

        return self.token_to_id[halt_symbol] if exists(halt_symbol) and halt_symbol in self.token_to_id else None

    @abstractmethod
    def encode(
        self,
        program: str
    ) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def decode(
        self,
        ids: list[int]
    ) -> str:
        raise NotImplementedError

    def __call__(
        self,
        program: str,
        input = '',
        seed = None
    ) -> str:
        return self.execute(program, input, seed).output

    def execute(
        self,
        program: str,
        input = '',
        seed = None
    ) -> ExecutionInfo:
        """output plus execution intermediates, falls back to wrapping an overridden `__call__`"""

        if type(self).__call__ is Executor.__call__:
            raise NotImplementedError

        return ExecutionInfo(program, self.__call__(program, input, seed), steps = len(program))
