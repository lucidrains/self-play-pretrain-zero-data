import random

from self_play_pretrain_zero_data import Brainfuck
from self_play_pretrain_zero_data.brainfuck import MACROS

HELLO_WORLD = (
    "++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]"
    ">>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++."
)


def test_hello_world():
    assert Brainfuck()(HELLO_WORLD) == "Hello World!\n"


def test_reads_input():
    assert Brainfuck()(',.', input='A') == 'A'


def test_macros_match_their_expansions():
    for macro, expansion in MACROS.items():
        for prefix in ('', '+', '+++', '+>+'):
            program = prefix + macro
            assert Brainfuck()(program) == Brainfuck()(program.replace(macro, expansion))


def test_macros_effects():
    assert Brainfuck()('+++++R>.') == '\x05'  # move value one cell right
    assert Brainfuck()('++++++++V') == '\x08'  # print until zero cell
    assert Brainfuck()('X.') == '\x10'  # set cell to 16


def test_unmatched_brackets_are_noops():
    assert Brainfuck()('+[+.') == '\x02'
    assert Brainfuck()('+.].') == '\x01\x01'


def test_circular_tape():
    # without wrapping the pointer would run off either end of the tape
    assert Brainfuck(tape_size=4)('>' * 4 + '+.') == '\x01'
    assert Brainfuck(tape_size=4)('<' * 5 + '+.') == '\x01'


def test_cells_wrap_modulo():
    assert Brainfuck()('-.') == '\xff'
    assert Brainfuck(modulus=3)('++++.') == '\x01'


def test_random_input_tape():
    # input string consumed first, then uniform random bytes
    assert Brainfuck()(',.,.', input='A')[:1] == 'A'
    assert Brainfuck()(',.', seed=0) == chr(random.Random(0).randrange(256))


def test_halts_at_f():
    assert Brainfuck()('+.F+.') == '\x01'


def test_halts_at_step_budget():
    assert Brainfuck(max_steps=100)('+[]') == ''


def test_halts_at_output_cap():
    assert Brainfuck(max_output_len=4)('+[.]') == '\x01' * 4


def test_every_string_is_executable():
    for program in ('[', ']', '[[', '?qZ', '+-<>[],.', ''):
        assert isinstance(Brainfuck()(program), str)
