import pytest
from self_play_pretrain_zero_data import Executor, SelfPlay


def test_self_play_executes_programs():
    class Echo(Executor):
        def __call__(self, program, **kwargs):
            return program

    self_play = SelfPlay(generator = None, learner = None, executor = Echo())

    assert self_play.execute('foo') == 'foo'
    assert self_play.execute(['foo', 'bar']) == ['foo', 'bar']

    with pytest.raises(NotImplementedError):
        self_play('foo')


def test_self_play_batch_seeds():
    class RecordSeed(Executor):
        def __call__(self, program, seed = None):
            return seed

    self_play = SelfPlay(generator = None, learner = None, executor = RecordSeed())

    assert self_play.execute(['a', 'b'], seed = 42) == [42, 43]
    assert self_play.execute(['a', 'b'], seed = [10, 20]) == [10, 20]

    with pytest.raises(ValueError):
        self_play.execute(['a', 'b'], seed = [10])

    with pytest.raises(AssertionError):
        self_play.execute('a', seed = [10])
