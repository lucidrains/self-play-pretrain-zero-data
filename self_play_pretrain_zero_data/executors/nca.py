from __future__ import annotations

from self_play_pretrain_zero_data.executors.base import (
    ExecutionInfo,
    Executor,
    input_stream,
)

# constants

# token alphabet - id 0 is reserved for sos / eos

DIGITS = '0123456789'
FRAME_SEPARATOR = ';'
ROW_SEPARATOR = '/'
FORWARD = '+'
BACKWARD = '-'
CYCLE = 'I'          # roll until a state repeats, i.e. to the initial state
INPUT_CELL = '?'
HALT = '!'

# von neumann neighborhood - updated cells only read the frozen opposite parity

NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))

# functions

def grid_key(grid):
    return tuple(tuple(row) for row in grid)

def parity_cells(height, width, parity):
    return ((row, col) for row in range(height) for col in range(width) if (row + col) % 2 == parity)

def neighborhood(grid, row, col):
    height, width = len(grid), len(grid[0])

    return tuple(grid[(row + dr) % height][(col + dc) % width] for dr, dc in NEIGHBORS)

def transition_index(index, offset, inverse):
    """transition index `offset` half steps before / after frame `index`"""
    return index - 1 - offset if inverse else index + offset

def infer_rule(frames, modulus):
    """recover the local rule from consecutive checkerboard half steps - first demo wins"""

    table = {}

    for index in range(len(frames) - 1):
        current, following = frames[index], frames[index + 1]

        for row, col in parity_cells(len(current), len(current[0]), index % 2):
            delta = (following[row][col] - current[row][col]) % modulus
            table.setdefault(neighborhood(current, row, col), delta)

    return table

def half_step(grid, table, parity, inverse, modulus):
    """one checkerboard half - every updated cell only reads the frozen opposite parity"""

    direction = -1 if inverse else 1

    for row, col in parity_cells(len(grid), len(grid[0]), parity):
        delta = table.get(neighborhood(grid, row, col), 0)
        grid[row][col] = (grid[row][col] + direction * delta) % modulus

    return grid

def roll(grid, table, index, depth, inverse, modulus):
    """half steps from transition `index` - forward counts up, backward counts down"""

    for offset in range(depth):
        half_step(grid, table, transition_index(index, offset, inverse) % 2, inverse, modulus)

    return grid

# classes

class NeuralCellularAutomata(Executor):
    """neural cellular automata with the rule inferred from context - every string is executable

    the local rule is latent - demonstrations of the dynamics are given as
    consecutive checkerboard half steps, the rule is recovered from the observed
    deltas and applied to the query frame, forwards or backwards in time

    `[+/-][depth or I] ; frame ; ... ; query`:
    - `+d` rolls the query `d` half steps forward, `-d` rolls it back in time
    - `I` rolls until a state repeats, outputting the initial state of the cycle
    - `?` reads the next input cell, uniform random cells by default

    after Training Language Models via Neural Cellular Automata (arxiv 2603.10055)
    """

    def __init__(
        self,
        num_colors = 10,     # cell states `n` in the paper
        max_steps = 64,      # rollout budget, then halt
        max_width = 64,      # frames are truncated past this width
        max_height = 64,     # frames are truncated past this height
        halt_symbol = HALT,  # program is truncated at this symbol
        random_input = True  # `?` falls back to uniform random cells, else zeros
    ):
        self.num_colors = min(max(num_colors, 1), len(DIGITS))
        self.max_steps = max_steps
        self.max_width = max(max_width, 0)
        self.max_height = max(max_height, 0)
        self.halt_symbol = halt_symbol
        self.random_input = random_input

        # tokenizer - id 0 is reserved for sos / eos

        alphabet = DIGITS + FRAME_SEPARATOR + ROW_SEPARATOR + FORWARD + BACKWARD + CYCLE + INPUT_CELL + (halt_symbol or '')

        self.alphabet = ''.join(dict.fromkeys(alphabet))
        self.num_tokens = len(self.alphabet) + 1
        self.token_to_id = {token: ind + 1 for ind, token in enumerate(self.alphabet)}
        self.id_to_token = {ind: token for token, ind in self.token_to_id.items()}

    def encode(self, program: str) -> list[int]:
        return [self.token_to_id[token] for token in program]

    def decode(self, ids: list[int]) -> str:
        return ''.join(self.id_to_token[i] for i in ids if i != self.sos_eos_id and i != self.pad_id)

    def execute(self, program: str, input = '', seed = None) -> ExecutionInfo:
        body = program.partition(self.halt_symbol)[0] if self.halt_symbol else program

        header, separator, frames_text = body.partition(FRAME_SEPARATOR)

        if not separator:
            header, frames_text = '', body

        inverse = BACKWARD in header
        cycle = CYCLE in header

        digits = ''.join(c for c in header if c in DIGITS)

        try:
            depth = min(int(digits or 1), self.max_steps)
        except ValueError:
            depth = self.max_steps  # int conversion limit

        reads = input_stream(input, self.num_colors, seed, self.random_input)

        frames = []

        for frame_text in frames_text.split(FRAME_SEPARATOR):
            rows = []

            for row_text in frame_text.split(ROW_SEPARATOR):
                row = [
                    next(reads) if c == INPUT_CELL else int(c)
                    for c in row_text
                    if c in DIGITS or c == INPUT_CELL
                ]

                if row:
                    rows.append([cell % self.num_colors for cell in row[:self.max_width]])

            if rows and self.max_height:
                frames.append(rows[:self.max_height])

        if not frames:
            return ExecutionInfo(program = program)

        query_lengths = [len(row) for row in frames[-1]]

        height = max(len(frame) for frame in frames)
        width = max(len(row) for frame in frames for row in frame)

        # even dimensions keep the checkerboard consistent around the torus

        height += height % 2
        width += width % 2

        frames = [
            [row + [0] * (width - len(row)) for row in frame]
            + [[0] * width for _ in range(height - len(frame))]
            for frame in frames
        ]

        table = infer_rule(frames, self.num_colors)

        grid = [row[:] for row in frames[-1]]
        query_index = len(frames) - 1

        seen = {grid_key(grid): 0}
        steps = loops = 0

        if cycle:
            while steps < self.max_steps:
                half_step(grid, table, transition_index(query_index, steps, inverse) % 2, inverse, self.num_colors)
                steps += 1

                key = grid_key(grid)

                if key in seen:
                    loops = steps - seen[key]
                    break

                seen[key] = steps
        else:
            steps = depth
            roll(grid, table, query_index, depth, inverse, self.num_colors)

        output = ROW_SEPARATOR.join(
            ''.join(map(str, grid[row][:length]))
            for row, length in enumerate(query_lengths)
        )

        return ExecutionInfo(
            program = program,
            output = output,
            steps = steps,
            loops = loops
        )
