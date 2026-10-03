import torch
from self_play_pretrain_zero_data import Brainfuck, SelfPlay
from self_play_pretrain_zero_data.self_play import Transformer

def test_transformer():
    ids = torch.randint(0, 256, (1, 1024))

    model = Transformer(num_tokens = 256, dim = 512, depth = 6)

    assert model(ids).shape == (1, 1024, 256)

def test_forward_with_jvp():
    model = Transformer(num_tokens = 256, dim = 64, depth = 2, dim_head = 16, heads = 4)

    ids = torch.randint(0, 256, (3, 17))

    tangent = {k: torch.randn_like(p) for k, p in model.named_parameters()}

    _, loss_tangent = model.forward_with_jvp(ids, tangent)

    assert loss_tangent.shape == (3,)

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
