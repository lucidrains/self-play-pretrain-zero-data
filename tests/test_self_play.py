import pytest
import torch
from torch.optim import AdamW, SGD

from torch_einops_utils import masked_mean

from self_play_pretrain_zero_data import Brainfuck, CheckpointReference, EMAReference, Executor, Forth, SelfPlay
from self_play_pretrain_zero_data.self_play import Transformer, char_decode, char_encode, exists, register_preconditioning, rewards_to_loss_weights

param = pytest.mark.parametrize

# swap this single param to test a different learner reference end to end

reference_fns = {
    'checkpoint': lambda tmp_path: CheckpointReference(folder = tmp_path),
    'ema': lambda _: EMAReference(decay = 0.99)
}

def test_transformer():
    ids = torch.randint(0, 256, (1, 1024))

    model = Transformer(num_tokens = 256, dim = 512, depth = 6)

    assert model(ids).shape == (1, 1024, 256)

    loss, loss_mask = model(ids, return_loss = True)

    assert loss.shape == loss_mask.shape == (1, 1023)

def test_transformer_weighted_loss():
    torch.manual_seed(0)

    model = Transformer(num_tokens = 256, dim = 16, depth = 1, dim_head = 8, heads = 2)

    ids = torch.randint(0, 256, (3, 17))
    rewards = torch.tensor([1., -1., 3.])

    weights = rewards_to_loss_weights(rewards)

    assert torch.allclose(weights, torch.tensor([0.25, 0., 0.75]))

    loss, loss_mask = model(ids, return_loss = True)
    seq_loss = masked_mean(loss, loss_mask, dim = -1)

    assert torch.allclose(model(ids, return_loss = True, reduce_loss = True), seq_loss.mean())

    weighted_loss = model(ids, return_loss = True, reduce_loss = True, loss_weights = weights)

    assert torch.allclose(weighted_loss, (seq_loss * weights).sum())

def test_empty_output_gets_eos_target():
    model = Transformer(num_tokens = 256 + 1, dim = 16, depth = 1, dim_head = 8, heads = 2)

    strings = ['', 'AB', chr(255)]
    ids = char_encode(strings)

    _, loss_mask = model(ids, return_loss = True)

    assert loss_mask[0, 0]
    assert not loss_mask[0, 1:].any()

    reduced_loss = model(ids, return_loss = True, reduce_loss = True)

    assert torch.isfinite(reduced_loss)

    assert [char_decode(row) for row in ids] == strings

def test_forward_with_jvp():
    model = Transformer(num_tokens = 256, dim = 64, depth = 2, dim_head = 16, heads = 4)

    ids = torch.randint(0, 256, (3, 17))

    tangent = {k: torch.randn_like(p) for k, p in model.named_parameters() if p.requires_grad}

    loss, loss_tangent = model.forward_with_jvp(ids, tangent)

    assert loss.shape == (3,)
    assert loss_tangent.shape == (3,)

class EmptyExecutor(Executor):
    num_tokens = 3

    def encode(self, program):
        return []

    def decode(self, ids):
        return ''

    def __call__(self, program, input = '', seed = None):
        return ''

def test_self_play_handles_empty_outputs(tmp_path):
    torch.manual_seed(0)

    executor = EmptyExecutor()

    generator = Transformer(num_tokens = executor.num_tokens, dim = 16, depth = 1, dim_head = 8, heads = 2)
    learner = Transformer(num_tokens = 256 + 1, dim = 16, depth = 1, dim_head = 8, heads = 2)

    self_play = SelfPlay(
        generator = generator,
        learner = learner,
        executor = executor,
        learner_reference = CheckpointReference(folder = tmp_path)
    )

    losses, tangents = self_play(batch_size = 2, max_length = 4, verbose = False, epochs = 2, decode_fn = executor.decode)

    assert torch.isfinite(losses).all()
    assert torch.isfinite(tangents).all()
    assert (losses > 0).all()
    assert (tangents != 0).any()

def test_generate_seq_mask_matches_derived_mask():
    torch.manual_seed(0)

    model = Transformer(num_tokens = 16, dim = 16, depth = 1, dim_head = 8, heads = 2)

    _, info = model.generate(batch_size = 4, max_length = 8, filter_thres = 0., return_for_policy_optimization = True)

    derived = info.decoded_ids != model.pad_id
    derived[:, :info.prompt_len] = False

    assert (derived == info.seq_mask).all()

@param('reference_fn', reference_fns.values(), ids = reference_fns.keys())
@param('executor_type', (Brainfuck, Forth))
@param('optimizer_type', (None, AdamW, SGD))
def test_self_play(reference_fn, optimizer_type, executor_type, tmp_path):
    torch.manual_seed(0)

    executor = executor_type()

    generator = Transformer(num_tokens = executor.num_tokens, dim = 32, depth = 1, dim_head = 8, heads = 4)
    learner = Transformer(num_tokens = 256 + 1, dim = 32, depth = 1, dim_head = 8, heads = 4)

    # sgd has no state to derive a preconditioner from, so just register a constant one

    register_preconditioning(SGD, lambda param, state, param_group: torch.full_like(param, param_group.get('lr', 1e-3)))

    # passing none uses the default learner optimizer, default learner tokenizer and default learner preconditioning

    learner_optimizer = optimizer_type(learner.parameters(), lr = 3e-4) if exists(optimizer_type) else None

    self_play = SelfPlay(
        generator = generator,
        learner = learner,
        executor = executor,
        learner_optimizer = learner_optimizer,
        learner_reference = reference_fn(tmp_path)
    )

    # generator samples programs, executes them, and the learner takes four steps on the encoded outputs
    # the executors expect decoded programs, so a decode fn is passed in

    num_epochs = 2

    losses, tangents = self_play(batch_size = 2, max_length = 8, verbose = False, epochs = num_epochs, decode_fn = executor.decode)

    assert losses.shape == (num_epochs,)
    assert tangents.shape == (num_epochs, 2)

    # the checkpoint reference persists an initial checkpoint plus one per epoch

    if isinstance(self_play.learner_reference, CheckpointReference):
        for epoch in range(num_epochs + 1):
            assert (tmp_path / f'learner.{epoch}.pt').exists()

    # parameter difference against the reference, and preconditioning for every gradient requiring learner parameter

    difference = self_play.parameter_difference()

    assert difference.keys() == learner.trainable_parameter_names()

    preconditioning = self_play.learner_preconditioning

    for name, param in learner.named_parameters():
        if param.requires_grad:
            assert preconditioning[name].shape == param.shape
