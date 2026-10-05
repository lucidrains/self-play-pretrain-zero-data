import random

from self_play_pretrain_zero_data import Brainfuck
from self_play_pretrain_zero_data.executors.brainfuck import MACROS

HELLO_WORLD = (
    "++++++++[>++++[>++>+++>+++>+<<<<-]>+>+>->>+[<]<-]"
    ">>.>---.+++++++..+++.>>.<-.<.+++.------.--------.>>+.>++."
)


def test_hello_world():
    assert Brainfuck()(HELLO_WORLD) == "Hello World!\n"


def test_execute_returns_intermediates():
    info = Brainfuck().execute('+++[-].')

    assert info.program == '+++[-].'
    assert info.output == '\x00'
    assert info.steps == 11
    assert info.loops == 2

    assert Brainfuck()('+++[-].') == info.output


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


def test_custom_macros():
    assert Brainfuck()('Q.') == '\x00'
    assert Brainfuck(macros = {'Q': '+++'})('Q.') == '\x03'
    assert Brainfuck(macros = {})('X.') == '\x00'


def test_unmatched_brackets_are_noops():
    assert Brainfuck()('+[+.') == '\x02'
    assert Brainfuck()('+.].') == '\x01\x01'


def test_unmatched_brackets_can_halt():
    assert Brainfuck()('[+.') == '\x01'
    assert Brainfuck(unmatched_brackets_are_noops=False)('[+.') == ''


def test_circular_tape():
    # without wrapping the pointer would run off either end of the tape
    assert Brainfuck(tape_size=4)('>' * 4 + '+.') == '\x01'
    assert Brainfuck(tape_size=4)('<' * 5 + '+.') == '\x01'


def test_tape_wrapping_can_be_disabled():
    program = '+++++++' + '<' * 5 + '>' * 5 + '.'
    assert Brainfuck(tape_size=4)(program) == '\x07'
    assert Brainfuck(tape_size=4, wrap_tape=False)(program) == '\x00'


def test_cells_wrap_modulo():
    assert Brainfuck()('-.') == '\xff'
    assert Brainfuck(cell_modulus=3)('++++.') == '\x01'


def test_random_input_tape():
    # input string consumed first, then uniform random bytes
    assert Brainfuck()(',.,.', input='A')[:1] == 'A'
    assert Brainfuck()(',.', seed=0) == chr(random.Random(0).randrange(256))


def test_input_fallback_can_be_zeros():
    assert Brainfuck(random_input=False)(',.', input='A') == 'A'
    assert Brainfuck(random_input=False)(',.') == '\x00'


def test_halts_at_f():
    assert Brainfuck()('+.F+.') == '\x01'


def test_halt_symbol_is_customizable():
    assert Brainfuck()('+.F+.') == '\x01'
    assert Brainfuck(halt_symbol='!')('+.F+.') == '\x01\x02'
    assert Brainfuck(halt_symbol=None)('+.F+.') == '\x01\x02'


def test_cell_modulus_larger_than_byte():
    assert Brainfuck(cell_modulus=512)('+' * 300 + '.') == chr(300 % 256)


def test_unicode_input():
    assert Brainfuck()(',.', input='世') == chr(ord('世') % 256)


def test_halts_at_step_budget():
    assert Brainfuck(max_steps=100)('+[]') == ''


def test_halts_at_output_cap():
    assert Brainfuck(max_output_len=4)('+[.]') == '\x01' * 4


def test_every_string_is_executable():
    for program in ('[', ']', '[[', '?qZ', '+-<>[],.', ''):
        assert isinstance(Brainfuck()(program), str)


def test_tokenizer():
    brainfuck = Brainfuck()

    assert brainfuck.sos_eos_id == 0
    assert brainfuck.num_tokens == len(brainfuck.alphabet) + 1
    assert brainfuck.decode([brainfuck.sos_eos_id]) == ''

    for program in (HELLO_WORLD, '+++.', 'ZR>+<.,[]F'):
        ids = brainfuck.encode(program)
        assert all(0 < i < brainfuck.num_tokens for i in ids)
        assert brainfuck.decode(ids) == program
