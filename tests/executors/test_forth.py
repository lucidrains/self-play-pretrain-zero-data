import random

from self_play_pretrain_zero_data.executors import Forth


def test_literals_and_arithmetic():
    assert Forth()('1 2 + EMIT') == '\x03'
    assert Forth()('10 3 - EMIT') == '\x07'
    assert Forth()('4 5 * EMIT') == '\x14'
    assert Forth()('5 NEGATE EMIT') == '\xfb'


def test_stack_words():
    assert Forth()('1 2 SWAP EMIT EMIT') == '\x01\x02'
    assert Forth()('65 DUP + EMIT') == '\x82'
    assert Forth()('1 2 DROP EMIT') == '\x01'
    assert Forth()('7 8 OVER EMIT EMIT EMIT') == '\x07\x08\x07'
    assert Forth()('1 2 3 ROT EMIT EMIT EMIT') == '\x01\x03\x02'


def test_zero_equals():
    assert Forth()('0 0= EMIT') == '\x01'
    assert Forth()('5 0= EMIT') == '\x00'


def test_cells_wrap_modulo():
    assert Forth()('256 EMIT') == '\x00'
    assert Forth()('-1 EMIT') == '\xff'
    assert Forth(cell_modulus=16)('20 EMIT') == '\x04'


def test_conditionals():
    assert Forth()('1 IF 65 EMIT ELSE 66 EMIT THEN') == 'A'
    assert Forth()('0 IF 65 EMIT ELSE 66 EMIT THEN') == 'B'
    assert Forth()('1 IF 65 EMIT THEN 67 EMIT') == 'AC'
    assert Forth()('0 IF 65 EMIT THEN 67 EMIT') == 'C'


def test_until_loop():
    assert Forth()('3 BEGIN DUP EMIT 1 - DUP 0= UNTIL') == '\x03\x02\x01'


def test_random_input_tape():
    # input string consumed first, then uniform random cells
    assert Forth()('KEY EMIT', input='A') == 'A'
    assert Forth()('KEY EMIT', seed=0) == chr(random.Random(0).randrange(256))


def test_input_fallback_can_be_zeros():
    assert Forth(random_input=False)('KEY EMIT') == '\x00'


def test_halts_at_halt_word():
    assert Forth()('65 EMIT BYE 66 EMIT') == 'A'
    assert Forth(halt_word='STOP')('65 EMIT STOP 66 EMIT') == 'A'
    assert Forth(halt_word=None)('65 EMIT BYE 66 EMIT') == 'AB'


def test_cell_modulus_larger_than_byte():
    assert Forth(cell_modulus=512)('300 EMIT') == chr(300 % 256)


def test_unicode_input():
    assert Forth()('KEY EMIT', input='世') == chr(ord('世') % 256)


def test_unknown_words_are_noops():
    assert Forth()('FOO 1 BAR EMIT') == '\x01'
    assert Forth(unknown_words_are_noops=False)('FOO 1 EMIT') == ''


def test_empty_stack_reads_zero():
    assert Forth()('EMIT') == '\x00'
    assert Forth(stack_underflow_is_zero=False)('EMIT') == ''


def test_stack_overflow_drops():
    assert Forth(stack_size=1)('1 2 EMIT EMIT') == '\x01\x00'


def test_case_insensitive_by_default():
    assert Forth()('1 2 + emit') == '\x03'
    assert Forth(case_sensitive=True)('1 2 + emit') == ''


def test_halts_at_step_budget():
    assert Forth(max_steps=100)('BEGIN 0 UNTIL') == ''


def test_halts_at_output_cap():
    assert Forth(max_output_len=4)('BEGIN 65 EMIT 0 UNTIL') == 'AAAA'


def test_every_string_is_executable():
    for program in ('IF', 'THEN', 'ELSE', 'UNTIL', 'BEGIN', '?q', ': foo ;', '', '9' * 5000 + ' EMIT'):
        assert isinstance(Forth()(program), str)
