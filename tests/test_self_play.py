import pytest
import torch
from torch.optim import AdamW, SGD

from self_play_pretrain_zero_data import Brainfuck, Forth, SelfPlay
from self_play_pretrain_zero_data.self_play import Transformer, exists, register_direction

param = pytest.mark.parametrize

@param('sos_eos_id', (None, 0))
def test_transformer(sos_eos_id):
    ids = torch.randint(0, 256, (1, 1024))

    model = Transformer(num_tokens = 256, dim = 512, depth = 6, sos_eos_id = sos_eos_id)

    auto_sos = int(sos_eos_id is not None)

    assert model(ids).shape == (1, 1024 + auto_sos, 256)

    loss, loss_mask = model(ids, return_loss = True)

    assert loss.shape == loss_mask.shape == (1, 1023 + auto_sos)

@param('sos_eos_id', (None, 0))
def test_forward_with_jvp(sos_eos_id):
    model = Transformer(num_tokens = 256, dim = 64, depth = 2, dim_head = 16, heads = 4, sos_eos_id = sos_eos_id)

    ids = torch.randint(0, 256, (3, 17))

    tangent = {k: torch.randn_like(p) for k, p in model.named_parameters()}

    loss, loss_tangent = model.forward_with_jvp(ids, tangent, detach_params = False)

    loss.backward()

    assert loss_tangent.shape == (3,)

@param('executor_type', (Brainfuck, Forth))
@param('optimizer_type', (None, AdamW, SGD))
def test_self_play(optimizer_type, executor_type):
    torch.manual_seed(0)

    executor = executor_type()

    generator = Transformer(num_tokens = executor.num_tokens, dim = 64, depth = 2, dim_head = 16, heads = 4, sos_eos_id = executor.sos_eos_id)
    learner = Transformer(num_tokens = 256, dim = 64, depth = 2, dim_head = 16, heads = 4)

    # sgd has no state to derive a direction from, so just register a constant one

    register_direction(SGD, lambda param, state: -torch.ones_like(param))

    # passing none uses the default learner optimizer, default learner tokenizer and default learner direction

    learner_optimizer = optimizer_type(learner.parameters(), lr = 3e-4) if exists(optimizer_type) else None

    self_play = SelfPlay(
        generator = generator,
        learner = learner,
        executor = executor,
        learner_optimizer = learner_optimizer
    )

    # generator samples programs, executes them, and the learner takes a step on the encoded outputs

    loss, tangent = self_play(batch_size = 2, max_length = 8, verbose = False)

    assert tangent.shape == (2,)
