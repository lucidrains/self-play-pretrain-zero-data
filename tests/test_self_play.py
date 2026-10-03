import pytest
from self_play_pretrain_zero_data import ProgramExecutor, SelfPlay


def test_self_play_executes_programs():
    class Echo(ProgramExecutor):
        def __call__(self, program, **kwargs):
            return program

    self_play = SelfPlay(generator = None, learner = None, executor = Echo())

    assert self_play.execute('foo') == 'foo'
    assert self_play.execute(['foo', 'bar']) == ['foo', 'bar']

    with pytest.raises(NotImplementedError):
        self_play('foo')
