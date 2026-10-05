from self_play_pretrain_zero_data.executors.base import (
    Executor,
    ExecutionInfo,
    input_stream,
    default
)

# constants

# single byte macro instructions from appendix E (table 4)

INSTRUCTIONS = '><+-.,[]'

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

# functions

def bracket_jumps(code):
    """bracket index -> paired bracket index, unmatched brackets omitted"""

    jumps, stack = {}, []

    for i, c in enumerate(code):
        if c == '[':
            stack.append(i)
        elif c == ']' and stack:
            left = stack.pop()
            jumps[left] = i
            jumps[i] = left

    return jumps

# classes

class Brainfuck(Executor):
    """appendix E execution semantics - every string is executable"""

    def __init__(
        self,
        tape_size = 1024,                     # number of cells
        cell_modulus = 256,                   # every cell is kept modulo this
        max_steps = 100_000,                  # instruction budget, then halt
        max_output_len = 1024,                # output byte budget, then halt
        halt_symbol = 'F',                    # code is truncated at this symbol, 'F' by default
        wrap_tape = True,                     # pointer wraps around the tape, else clamps at either end
        unmatched_brackets_are_noops = True,  # active stray brackets are ignored, else halt
        random_input = True,                  # ',' falls back to uniform random cells, else zeros
        macros = None                         # symbol -> expansion table, default MACROS (appendix E table 4)
    ):
        self.tape_size = max(tape_size, 1)
        self.cell_modulus = max(cell_modulus, 1)
        self.max_steps = max_steps
        self.max_output_len = max_output_len
        self.halt_symbol = halt_symbol
        self.wrap_tape = wrap_tape
        self.unmatched_brackets_are_noops = unmatched_brackets_are_noops
        self.random_input = random_input
        self.macros = default(macros, MACROS)

        # tokenizer - id 0 is reserved for sos / eos

        alphabet = INSTRUCTIONS + ''.join(self.macros)
        if self.halt_symbol:
            alphabet += self.halt_symbol

        self.alphabet = ''.join(dict.fromkeys(alphabet))
        self.num_tokens = len(self.alphabet) + 1
        self.token_to_id = {token: ind + 1 for ind, token in enumerate(self.alphabet)}
        self.id_to_token = {ind: token for token, ind in self.token_to_id.items()}

    def encode(self, program: str) -> list[int]:
        return [self.token_to_id[token] for token in program]

    def decode(self, ids: list[int]) -> str:
        return ''.join(self.id_to_token[i] for i in ids if i != self.sos_eos_id and i != self.pad_id)

    def move(self, ptr: int, step: int) -> int:
        if self.wrap_tape:
            return (ptr + step) % self.tape_size
        return max(0, min(ptr + step, self.tape_size - 1))

    def execute(self, program: str, input = '', seed = None) -> ExecutionInfo:
        body = program.partition(self.halt_symbol)[0] if self.halt_symbol else program
        code = ''.join(self.macros.get(c, c) for c in body)
        jumps = bracket_jumps(code)

        reads = input_stream(input, self.cell_modulus, seed, self.random_input)

        tape = [0] * self.tape_size
        out = bytearray()
        pc = ptr = steps = loops = 0

        for _ in range(self.max_steps):
            if pc >= len(code) or len(out) >= self.max_output_len:
                break

            steps += 1

            match code[pc]:
                case '>': ptr = self.move(ptr, 1)
                case '<': ptr = self.move(ptr, -1)
                case '+': tape[ptr] = (tape[ptr] + 1) % self.cell_modulus
                case '-': tape[ptr] = (tape[ptr] - 1) % self.cell_modulus
                case '.': out.append(tape[ptr] % 256)
                case ',': tape[ptr] = next(reads)
                case '[' if not tape[ptr]:
                    if pc in jumps:
                        pc = jumps[pc]
                    elif not self.unmatched_brackets_are_noops:
                        break
                case ']' if tape[ptr]:
                    if pc in jumps:
                        pc = jumps[pc]
                        loops += 1
                    elif not self.unmatched_brackets_are_noops:
                        break

            pc += 1

        return ExecutionInfo(
            program = program,
            output = out.decode('latin-1'),
            steps = steps,
            loops = loops
        )
