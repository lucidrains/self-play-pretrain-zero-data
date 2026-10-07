import math
from importlib.util import find_spec

import pytest
import torch
from torch.optim import AdamW, SGD

from torch_einops_utils import masked_mean

from self_play_pretrain_zero_data import Brainfuck, CheckpointReference, EMAReference, Executor, Forth, NeuralCellularAutomata, SelfPlay
from self_play_pretrain_zero_data.self_play import HAS_JVP_FLASH_ATTENTION, Transformer, char_decode, char_encode, exists, register_preconditioning, representation_alignment_loss, rewards_to_loss_weights

HAS_TRITON = find_spec('triton') is not None

param = pytest.mark.parametrize

requires_jvp_flash = pytest.mark.skipif(
    not (HAS_JVP_FLASH_ATTENTION and HAS_TRITON and torch.cuda.is_available()),
    reason = 'jvp flash attention requires triton and cuda'
)

jvp_flash_attn = pytest.param('jvp_flash', marks = requires_jvp_flash)

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

def test_parallel_forward_matches_stepwise_memory():
    torch.manual_seed(0)

    model = Transformer(num_tokens = 256, dim = 64, depth = 3, dim_head = 16, heads = 4).eval()

    ids = torch.randint(0, 256, (2, 8))

    with torch.no_grad():
        parallel = model(ids)

        memory, stepwise = None, []

        for token in ids.split(1, dim = 1):  # one token at a time, carrying memory
            logits, memory = model(token, memory = memory, return_memory = True)
            stepwise.append(logits)

    assert torch.allclose(parallel, torch.cat(stepwise, dim = 1), atol = 1e-5)

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

@param('attn_type', ('plain', 'sdpa', jvp_flash_attn))
def test_attention_types(attn_type):
    torch.manual_seed(0)

    ids = torch.randint(0, 256, (2, 32))

    plain = Transformer(num_tokens = 256, dim = 32, depth = 2, dim_head = 16, heads = 4, attn_type = 'plain')

    expected = plain(ids)

    model = Transformer(num_tokens = 256, dim = 32, depth = 2, dim_head = 16, heads = 4, attn_type = attn_type)
    model.load_state_dict(plain.state_dict())

    logits = model(ids)

    assert logits.shape == expected.shape == (2, 32, 256)
    assert torch.allclose(logits, expected, atol = 1e-5)

    loss, loss_mask = model(ids, return_loss = True)

    assert loss.shape == loss_mask.shape == (2, 31)

@param('attn_type', ('plain', jvp_flash_attn))
def test_forward_with_jvp(attn_type):
    torch.manual_seed(0)

    ids = torch.randint(0, 256, (3, 33))  # one shifted away for the next token loss, leaving 32

    plain = Transformer(num_tokens = 256, dim = 64, depth = 2, dim_head = 16, heads = 4, attn_type = 'plain')

    tangent = {k: torch.randn_like(p) for k, p in plain.named_parameters() if p.requires_grad}

    expected_loss, expected_tangent = plain.forward_with_jvp(ids, tangent)

    model = Transformer(num_tokens = 256, dim = 64, depth = 2, dim_head = 16, heads = 4, attn_type = attn_type)
    model.load_state_dict(plain.state_dict())

    loss, loss_tangent = model.forward_with_jvp(ids, tangent)

    assert loss.shape == loss_tangent.shape == (3,)
    assert torch.allclose(loss, expected_loss, atol = 1e-5)
    assert torch.allclose(loss_tangent, expected_tangent, atol = 1e-5)

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

def test_representation_alignment(tmp_path):
    torch.manual_seed(0)

    model = Transformer(num_tokens = 256, dim = 8, depth = 1, dim_head = 8, heads = 1)

    ids = torch.randint(1, 256, (4, 12))

    logits, pooled_repr = model(ids, return_pooled_repr = True)

    assert logits.shape == (4, 12, 256)
    assert pooled_repr.shape == (4, 8)

    loss, pooled_repr = model(ids, return_loss = True, reduce_loss = True, return_pooled_repr = True)

    assert loss.shape == ()
    assert pooled_repr.shape == (4, 8)

    # cosine alignment against a detached generator target, zero for identical representations

    generator_repr = torch.randn(4, 8)

    alignment_loss = representation_alignment_loss(pooled_repr, generator_repr.detach())

    expected = (1. - torch.nn.functional.cosine_similarity(pooled_repr, generator_repr, dim = -1)).mean()

    assert torch.allclose(alignment_loss, expected)

    alignment_loss.backward()

    assert model.token_emb.weight.grad is not None

    same = torch.randn(3, 8)

    assert representation_alignment_loss(same, same.detach()).item() < 1e-6

    # end to end with the alignment term active

    executor = EmptyExecutor()

    generator = Transformer(num_tokens = executor.num_tokens, dim = 8, depth = 1, dim_head = 8, heads = 1)
    learner = Transformer(num_tokens = 256 + 1, dim = 8, depth = 1, dim_head = 8, heads = 1)

    self_play = SelfPlay(
        generator = generator,
        learner = learner,
        executor = executor,
        learner_reference = CheckpointReference(folder = tmp_path),
        representation_alignment_loss_weight = 1.
    )

    losses, tangents = self_play(batch_size = 2, max_length = 4, verbose = False, epochs = 1, decode_fn = executor.decode)

    assert torch.isfinite(losses).all()
    assert torch.isfinite(tangents).all()

def test_generate_seq_mask_matches_derived_mask():
    torch.manual_seed(0)

    model = Transformer(num_tokens = 16, dim = 16, depth = 1, dim_head = 8, heads = 2)

    generate_kwargs = (
        dict(batch_size = 4),
        dict(prompt_ids = [[3, 4], [5, 6]], prepend_sos = True)
    )

    for kwargs in generate_kwargs:
        _, info = model.generate(max_length = 8, filter_thres = 0., return_for_policy_optimization = True, **kwargs)

        # fallback derived by grpo_loss when seq_mask is not given
        derived = info.decoded_ids != model.pad_id
        derived[:, :info.prompt_len] = False

        assert (derived == info.seq_mask).all()

def test_generate_seq_mask_includes_terminator():
    torch.manual_seed(0)

    model = Transformer(num_tokens = 16, dim = 16, depth = 1, dim_head = 8, heads = 2)

    halt_id = 3

    def only_halt(logits, _):
        mask = torch.arange(logits.shape[-1], device = logits.device) == halt_id
        return logits.masked_fill(~mask, -torch.finfo(logits.dtype).max)

    _, info = model.generate(
        batch_size = 2,
        max_length = 4,
        filter_logits_fn = only_halt,
        filter_thres = 1.,
        eos_ids = (halt_id,),
        mask_out_eos_on_first_step = False,
        return_for_policy_optimization = True
    )

    # the terminator is the last content token, giving a program log prob up to and including it

    assert (info.decoded_ids[:, info.prompt_len] == halt_id).all()
    assert info.seq_mask[:, info.prompt_len].all()
    assert (info.seq_mask.sum(dim = -1) == 1).all()

def test_uniform_prior_uses_program_alphabet_size(tmp_path):
    torch.manual_seed(0)

    executor = Brainfuck()

    generator = Transformer(num_tokens = executor.num_tokens, dim = 16, depth = 1, dim_head = 8, heads = 2)
    learner = Transformer(num_tokens = 256 + 1, dim = 16, depth = 1, dim_head = 8, heads = 2)

    self_play = SelfPlay(
        generator = generator,
        learner = learner,
        executor = executor,
        learner_reference = CheckpointReference(folder = tmp_path)
    )

    # a terminated three token program, terminator included in the mask

    seq_mask = torch.tensor([[True, True, True, False]])
    replay_ids = torch.tensor([[1, 2, executor.halt_id, -1]])

    log_prob = self_play.prior_log_probs(replay_ids, seq_mask)

    alphabet_size = len(executor.alphabet)

    assert alphabet_size == executor.num_tokens - 1
    assert torch.allclose(log_prob, torch.tensor([-3. * math.log(alphabet_size)]))

@param('reference_fn', reference_fns.values(), ids = reference_fns.keys())
@param('executor_type', (Brainfuck, Forth, NeuralCellularAutomata))
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
