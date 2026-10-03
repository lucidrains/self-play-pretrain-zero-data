from __future__ import annotations

import random
from itertools import chain, repeat

from self_play_pretrain_zero_data.self_play import ProgramExecutor

# single byte macro instructions from appendix E (table 4)

MACROS = {
    'Z': '[-]',
    'R': '[->+<]',
    'L': '[->+++<]',
    'N': '[-<->]',
    'C': '[->+>+<<]',
    'G': '[>]',
    'H': '[<]',
    'W': '[[-]>+<]',
    'V': '[.>]',
    'X': '[-]++++++++++++++++',
}


def bracket_jumps(code):
    jumps, stack = {}, []

    for i, c in enumerate(code):
        if c == '[':
            stack.append(i)
        elif c == ']' and stack:
            left = stack.pop()
            jumps[left] = i
            jumps[i] = left

    return jumps

class Brainfuck(ProgramExecutor):
    """appendix E execution semantics - every string is executable

    - circular tape, cells modulo `modulus`
    - unmatched brackets are no-ops
    - `,` reads the input string, then uniform random bytes (random input tape omega)
    - halts at `F`, the step budget, or the output cap
    """

    def __init__(
        self,
        tape_size = 1024,
        modulus = 256,
        max_steps = 100_000,
        max_output_len = 1024
    ):
        self.tape_size = tape_size
        self.modulus = modulus
        self.max_steps = max_steps
        self.max_output_len = max_output_len

    def __call__(
        self,
        program: str,
        input: str | bytes = '',
        seed: int | None = None
    ) -> str:
        rng = random.Random(seed)

        code = ''.join(MACROS.get(c, c) for c in program.partition('F')[0])
        jumps = bracket_jumps(code)

        input_bytes = input.encode('latin-1') if isinstance(input, str) else input

        reads = chain(
            (b % self.modulus for b in input_bytes),
            (rng.randrange(self.modulus) for _ in repeat(None))
        )

        tape = [0] * self.tape_size
        out = bytearray()
        pc = ptr = 0

        for _ in range(self.max_steps):
            if pc >= len(code) or len(out) >= self.max_output_len:
                break

            match code[pc]:
                case '>': ptr = (ptr + 1) % self.tape_size
                case '<': ptr = (ptr - 1) % self.tape_size
                case '+': tape[ptr] = (tape[ptr] + 1) % self.modulus
                case '-': tape[ptr] = (tape[ptr] - 1) % self.modulus
                case '.': out.append(tape[ptr])
                case ',': tape[ptr] = next(reads)
                case '[' if not tape[ptr]: pc = jumps.get(pc, pc)
                case ']' if tape[ptr]: pc = jumps.get(pc, pc)

            pc += 1

        return out.decode('latin-1')
