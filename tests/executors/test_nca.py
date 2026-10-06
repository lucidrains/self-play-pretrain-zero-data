import random

from self_play_pretrain_zero_data import NeuralCellularAutomata
from self_play_pretrain_zero_data.executors.nca import half_step, infer_rule


def serialize(grid):
    return '/'.join(''.join(str(cell) for cell in row) for row in grid)


def test_forward_rollout():
    program = '+1;11/11;21/12;22/22'

    info = NeuralCellularAutomata().execute(program)

    assert info.output == '32/23'
    assert info.steps == 1
    assert info.loops == 0

    assert NeuralCellularAutomata()(program) == info.output


def test_backward_rollout():
    info = NeuralCellularAutomata().execute('-1;11/11;21/12;22/22')

    assert info.output == '21/12'
    assert info.steps == 1


def test_backward_undoes_forward():
    rng = random.Random(0)
    table = {
        (vertical, vertical, horizontal, horizontal): rng.randrange(10)
        for vertical in range(10)
        for horizontal in range(10)
    }

    grids = [[[1, 2], [3, 4]]]

    for index in range(2):
        following = [row[:] for row in grids[-1]]
        half_step(following, table, index % 2, inverse = False, modulus = 10)
        grids.append(following)

    demos = ';'.join(serialize(grid) for grid in grids)
    forward = NeuralCellularAutomata()('+1;' + demos)
    backward = NeuralCellularAutomata()('-1;' + demos + ';' + forward)

    assert backward == serialize(grids[-1])


def test_first_demo_wins():
    frames = [
        [[1, 1], [1, 1]],
        [[2, 1], [1, 2]],
        [[2, 1], [1, 2]],
        [[4, 1], [1, 4]],
    ]

    table = infer_rule(frames, 10)

    assert table[(1, 1, 1, 1)] == 1


def test_no_demos_keeps_the_query():
    info = NeuralCellularAutomata().execute('12/34')

    assert info.output == '12/34'
    assert info.steps == 1


def test_depth_zero():
    info = NeuralCellularAutomata().execute('+0;12/34')

    assert info.output == '12/34'
    assert info.steps == 0


def test_rewinds_to_the_initial_state():
    info = NeuralCellularAutomata().execute('-I;12/34')

    assert info.output == '12/34'
    assert info.steps == 1
    assert info.loops == 1


def test_input_cells():
    assert NeuralCellularAutomata()('+0;??', input = '\x03\x04') == '34'


def test_random_input_cells():
    rng = random.Random(0)
    expected = ''.join(str(rng.randrange(10)) for _ in range(3))

    assert NeuralCellularAutomata()('+0;???', seed = 0) == expected


def test_input_fallback_can_be_zeros():
    assert NeuralCellularAutomata(random_input = False)('+0;??') == '00'


def test_halts_at_bang_symbol():
    assert NeuralCellularAutomata()('+1;11/11;21/12;22/22!99') == NeuralCellularAutomata()('+1;11/11;21/12;22/22')


def test_odd_frames_are_padded_but_cropped():
    assert NeuralCellularAutomata()('+0;123/456/789') == '123/456/789'


def test_ragged_rows_are_kept():
    assert NeuralCellularAutomata()('+0;345/6') == '345/6'


def test_frames_are_truncated():
    assert NeuralCellularAutomata(max_width = 2, max_height = 1)('+0;1234/5678') == '12'
    assert NeuralCellularAutomata(max_height = 0)('+0;12/34') == ''


def test_cells_wrap_modulo_the_color_count():
    assert NeuralCellularAutomata(num_colors = 2)('+0;9') == '1'


def test_empty_program():
    for program in ('', ';', ';;'):
        info = NeuralCellularAutomata().execute(program)

        assert info.output == ''
        assert info.steps == 0
        assert info.loops == 0


def test_every_string_is_executable():
    for program in ('', ';', ';;;', '?/?', 'hello', 'I', '-I/', '0' * 500, '12/34/56;+' * 10, '+' + '9' * 5000 + ';12/34'):
        assert isinstance(NeuralCellularAutomata()(program), str)


def test_tokenizer():
    nca = NeuralCellularAutomata()

    assert nca.sos_eos_id == 0
    assert nca.num_tokens == len(nca.alphabet) + 1
    assert nca.decode([nca.sos_eos_id]) == ''

    for program in ('+1;11/11;21/12;22/22!', '-I;12/34', '?/?;0123456789'):
        ids = nca.encode(program)
        assert all(0 < i < nca.num_tokens for i in ids)
        assert nca.decode(ids) == program
