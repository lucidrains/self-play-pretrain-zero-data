import pytest
import torch
from torch.optim import AdamW, SGD

from self_play_pretrain_zero_data import Brainfuck, Forth, SelfPlay
from self_play_pretrain_zero_data.self_play import Transformer, exists, register_preconditioning

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

    tangent = {k: torch.randn_like(p) for k, p in model.named_parameters() if p.requires_grad}

    loss, loss_tangent = model.forward_with_jvp(ids, tangent)

    assert loss.shape == ()
    assert loss_tangent.shape == (3,)

@param('executor_type', (Brainfuck, Forth))
@param('optimizer_type', (None, AdamW, SGD))
def test_self_play(optimizer_type, executor_type, tmp_path):
    torch.manual_seed(0)

    executor = executor_type()

    generator = Transformer(num_tokens = executor.num_tokens, dim = 32, depth = 1, dim_head = 8, heads = 4, sos_eos_id = executor.sos_eos_id)
    learner = Transformer(num_tokens = 256, dim = 32, depth = 1, dim_head = 8, heads = 4)

    # sgd has no state to derive a preconditioner from, so just register a constant one

    register_preconditioning(SGD, lambda param, state, param_group: torch.full_like(param, param_group.get('lr', 1e-3)))

    # passing none uses the default learner optimizer, default learner tokenizer and default learner preconditioning

    learner_optimizer = optimizer_type(learner.parameters(), lr = 3e-4) if exists(optimizer_type) else None

    self_play = SelfPlay(
        generator = generator,
        learner = learner,
        executor = executor,
        learner_optimizer = learner_optimizer,
        learner_checkpoint_folder = tmp_path
    )

    # generator samples programs, executes them, and the learner takes four steps on the encoded outputs
    # the executors expect decoded programs, so a decode fn is passed in

    num_epochs = 2

    losses, tangents = self_play(batch_size = 2, max_length = 8, verbose = False, num_epochs = num_epochs, decode_fn = executor.decode)

    assert losses.shape == (num_epochs,)
    assert tangents.shape == (num_epochs, 2)

    # initial checkpoint plus one per epoch

    for epoch in range(num_epochs + 1):
        assert (tmp_path / f'learner.{epoch}.pt').exists()

    # parameter difference against the initial checkpoint, and preconditioning for every gradient requiring learner parameter

    difference = self_play.parameter_difference(0)

    assert difference.keys() == learner.trainable_parameter_names()

    preconditioning = self_play.learner_preconditioning

    for name, param in learner.named_parameters():
        if param.requires_grad:
            assert preconditioning[name].shape == param.shape
