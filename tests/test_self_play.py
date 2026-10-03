import pytest
import torch
from torch.optim import AdamW, SGD

from self_play_pretrain_zero_data import Brainfuck, SelfPlay
from self_play_pretrain_zero_data.self_play import Transformer, exists

@pytest.mark.parametrize('sos_eos_id', (None, 0))
def test_transformer(sos_eos_id):
    ids = torch.randint(0, 256, (1, 1024))

    model = Transformer(num_tokens = 256, dim = 512, depth = 6, sos_eos_id = sos_eos_id)

    auto_sos = int(sos_eos_id is not None)

    assert model(ids).shape == (1, 1024 + auto_sos, 256)

    loss, loss_mask = model(ids, return_loss = True)

    assert loss.shape == loss_mask.shape == (1, 1023 + auto_sos)

@pytest.mark.parametrize('sos_eos_id', (None, 0))
def test_forward_with_jvp(sos_eos_id):
    model = Transformer(num_tokens = 256, dim = 64, depth = 2, dim_head = 16, heads = 4, sos_eos_id = sos_eos_id)

    ids = torch.randint(0, 256, (3, 17))

    tangent = {k: torch.randn_like(p) for k, p in model.named_parameters()}

    loss, loss_tangent = model.forward_with_jvp(ids, tangent, detach_params = False)

    loss.backward()

    assert loss_tangent.shape == (3,)

def makeshift_direction(param, state):
    momentum = state.get('momentum_buffer')

    if not exists(momentum):
        return -torch.ones_like(param)

    return -momentum

@pytest.mark.parametrize('optimizer_type', (None, AdamW, SGD))
def test_self_play(optimizer_type):
    torch.manual_seed(0)

    brainfuck = Brainfuck()

    generator = Transformer(num_tokens = brainfuck.num_tokens, dim = 64, depth = 2, dim_head = 16, heads = 4, sos_eos_id = brainfuck.sos_eos_id)
    learner = Transformer(num_tokens = 256, dim = 64, depth = 2, dim_head = 16, heads = 4)

    # passing none uses the default learner optimizer, default learner tokenizer and default learner direction

    learner_optimizer, learner_direction_fn = None, None

    if exists(optimizer_type):
        learner_optimizer = optimizer_type(learner.parameters(), lr = 3e-4)
        learner_direction_fn = None if optimizer_type is AdamW else makeshift_direction

    self_play = SelfPlay(
        generator = generator,
        learner = learner,
        executor = brainfuck,
        learner_optimizer = learner_optimizer,
        learner_direction_fn = learner_direction_fn
    )

    # generator samples programs, executes them, and the learner takes a step on the encoded outputs

    loss, tangent = self_play(batch_size = 2, max_length = 8, verbose = False)

    assert tangent.shape == (2,)
