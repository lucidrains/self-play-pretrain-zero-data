from __future__ import annotations

import random
from abc import ABC, abstractmethod
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

    @abstractmethod
    def __call__(
        self,
        program: str,
        input = '',
        seed = None
    ) -> str:
        raise NotImplementedError
