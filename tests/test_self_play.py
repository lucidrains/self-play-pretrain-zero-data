import torch
from self_play_pretrain_zero_data import Brainfuck, SelfPlay
from self_play_pretrain_zero_data.self_play import Transformer

def test_transformer():
    ids = torch.randint(0, 256, (1, 1024))

    model = Transformer(num_tokens = 256, dim = 512, depth = 6)

    assert model(ids).shape == (1, 1024, 256)

def test_self_play():

    brainfuck = Brainfuck()

    learner = Transformer(num_tokens = 256, dim = 512, depth = 6)
    generator = Transformer(num_tokens = 256, dim = 512, depth = 6)

    self_play = SelfPlay(
        generator = generator,
        learner = learner,
        executor = brainfuck
    )

    assert self_play.execute_programs(['+++.']) == ['\x03']
