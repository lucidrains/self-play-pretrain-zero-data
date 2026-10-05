from __future__ import annotations

import re
import string

from self_play_pretrain_zero_data.executors.base import Executor, ExecutionInfo, input_stream

# constants

# minimal total forth - a stack machine with structured control flow

INTEGER = re.compile(r'[+-]?\d+')

# exceptions

class _Halt(Exception):
    pass

# functions

def control_jumps(tokens):
    """token index -> paired control word index, unmatched words omitted"""

    jumps = {}
    ifs, begins = [], []

    for i, token in enumerate(tokens):
        if token == 'IF':
            ifs.append(i)
        elif token == 'ELSE' and ifs:
            jumps[ifs[-1]] = i
            ifs[-1] = i  # the else branch is now the open if
        elif token == 'THEN' and ifs:
            jumps[ifs.pop()] = i
        elif token == 'BEGIN':
            begins.append(i)
        elif token == 'UNTIL' and begins:
            jumps[i] = begins.pop()

    return jumps

# classes

class Forth(Executor):
    """minimal forth execution semantics - every string is executable

    words - literals, `DUP DROP SWAP OVER ROT`, `+ - * NEGATE 0=`, `EMIT KEY`,
    `IF ELSE THEN` and `BEGIN UNTIL` control flow, unmatched words are no-ops
    """

    def __init__(
        self,
        stack_size = 1024,               # number of cells, extra pushes are dropped
        cell_modulus = 256,              # every cell is kept modulo this
        max_steps = 100_000,             # token budget (each word or literal costs 1 step), then halt
        max_output_len = 1024,           # output byte budget, then halt
        halt_word = 'BYE',               # tokens from here on are ignored, 'BYE' by default
        unknown_words_are_noops = True,  # unknown tokens are ignored, else halt
        stack_underflow_is_zero = True,  # empty stack pops yield zero, else halt
        random_input = True,             # KEY falls back to uniform random cells, else zeros
        case_sensitive = False           # tokens are folded to uppercase by default
    ):
        self.stack_size = max(stack_size, 0)
        self.cell_modulus = max(cell_modulus, 1)
        self.max_steps = max_steps
        self.max_output_len = max_output_len
        self.halt_word = halt_word
        self.unknown_words_are_noops = unknown_words_are_noops
        self.stack_underflow_is_zero = stack_underflow_is_zero
        self.random_input = random_input
        self.case_sensitive = case_sensitive

        # tokenizer - char level, id 0 is reserved for sos / eos

        alphabet = string.ascii_uppercase + string.ascii_lowercase + string.digits + ' ' + '+-*='

        self.alphabet = ''.join(dict.fromkeys(alphabet))
        self.num_tokens = len(self.alphabet) + 1
        self.token_to_id = {token: ind + 1 for ind, token in enumerate(self.alphabet)}
        self.id_to_token = {ind: token for token, ind in self.token_to_id.items()}

    def encode(self, program: str) -> list[int]:
        return [self.token_to_id[token] for token in program]

    def decode(self, ids: list[int]) -> str:
        return ''.join(self.id_to_token[i] for i in ids if i != self.sos_eos_id and i != self.pad_id)

    def execute(self, program: str, input = '', seed = None) -> ExecutionInfo:
        tokens = program.split() if self.case_sensitive else program.upper().split()

        halt_word = self.halt_word if self.case_sensitive or not self.halt_word else self.halt_word.upper()
        if halt_word and halt_word in tokens:
            tokens = tokens[:tokens.index(halt_word)]

        jumps = control_jumps(tokens)
        reads = input_stream(input, self.cell_modulus, seed, self.random_input)

        stack, out = [], bytearray()
        pc = steps = loops = 0

        def pop():
            if stack:
                return stack.pop()
            if self.stack_underflow_is_zero:
                return 0
            raise _Halt

        def push(*values):
            for v in values:
                if len(stack) < self.stack_size:
                    stack.append(v % self.cell_modulus)

        try:
            for _ in range(self.max_steps):
                if pc >= len(tokens) or len(out) >= self.max_output_len:
                    break

                steps += 1
                token = tokens[pc]

                match token:
                    case 'DUP':
                        a = pop()
                        push(a, a)
                    case 'DROP':
                        pop()
                    case 'SWAP':
                        b, a = pop(), pop()
                        push(b, a)
                    case 'OVER':
                        b, a = pop(), pop()
                        push(a, b, a)
                    case 'ROT':
                        c, b, a = pop(), pop(), pop()
                        push(b, c, a)
                    case '+':
                        b, a = pop(), pop()
                        push(a + b)
                    case '-':
                        b, a = pop(), pop()
                        push(a - b)
                    case '*':
                        b, a = pop(), pop()
                        push(a * b)
                    case 'NEGATE':
                        push(-pop())
                    case '0=':
                        push(int(pop() == 0))
                    case 'EMIT':
                        out.append(pop() % 256)
                    case 'KEY':
                        push(next(reads))
                    case 'IF':
                        if pop() == 0:
                            pc = jumps.get(pc, pc)
                    case 'UNTIL':
                        if pop() == 0 and pc in jumps:
                            pc = jumps[pc]
                            loops += 1
                    case 'ELSE':
                        pc = jumps.get(pc, pc)
                    case 'THEN' | 'BEGIN':
                        pass
                    case _ if INTEGER.fullmatch(token):
                        try:
                            push(int(token))
                        except ValueError:
                            pass  # integer string conversion limit (>4300 digits, CVE-2020-10735)
                    case _:
                        if not self.unknown_words_are_noops:
                            break

                pc += 1
        except _Halt:
            pass

        return ExecutionInfo(
            program = program,
            output = out.decode('latin-1'),
            steps = steps,
            loops = loops
        )
