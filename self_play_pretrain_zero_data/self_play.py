from __future__ import annotations
from types import SimpleNamespace
from copy import deepcopy

from functools import partial
from math import ceil
from pathlib import Path

import math

import torch
import torch.nn.functional as F
from torch import nn, arange, cat, isin, stack, tensor, is_tensor
from torch.nn import Module, ModuleList, Linear
from torch.optim import Adam, AdamW
from torch.func import functional_call, jvp

from accelerate import Accelerator

from ema_pytorch import EMA

from einops import einsum, rearrange
from einops.layers.torch import Rearrange

from torch_einops_utils import batched_index_select, clamp, lens_to_mask, masked_mean, masked_sum, pack_with_inverse, pad_left_at_dim, pad_sequence, temp_eval, tree_map_detach, z_score
from torch_einops_utils.device import move_inputs_to_module_device
from torch_einops_utils.shape import shape, size
from torch_einops_utils.torch_einops_utils import identity

from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

from self_play_pretrain_zero_data.executors import Executor

# constants

LinearNoBias = partial(Linear, bias = False)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def pick(d, keys):
    return tuple(d[key] for key in keys)

# rewards to expert iteration weights - section 2.2 of the paper, w_i in the generator objective (eq. 5)

def rewards_to_loss_weights(
    rewards, # (b,)
    eps = 1e-5
):
    # w_i = [r_i]_+ / sum_j [r_j]_+

    positive_rewards = F.relu(rewards)
    total_positive_reward = positive_rewards.sum().clamp_min(eps)

    return positive_rewards / total_positive_reward

# sampling helpers

def log(t, eps = 1e-20):
    return t.clamp_min(eps).log()

def gumbel_noise_like(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))

def top_k(logits, thres = 0.9):
    num_tokens = size(logits, '... [v]')

    k = clamp(ceil((1. - thres) * num_tokens), lo = 1, hi = num_tokens)

    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(-1, ind, val)

    return probs

def gumbel_sample(logits, temperature = 1.):
    logits = logits / temperature
    return (logits + gumbel_noise_like(logits)).argmax(dim = -1)

# rmsnorm, as the fused torch kernel is not forward ad compatible on mps

class RMSNorm(Module):
    def __init__(
        self,
        dim,
        eps = 1e-8
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim = -1, keepdim = True) + self.eps) * self.weight

# attention

class Attention(Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        self.norm = RMSNorm(dim)
        self.to_qkv = LinearNoBias(dim, dim_inner * 3)
        self.to_out = LinearNoBias(dim_inner, dim)

        self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

    def forward(
        self,
        x,
        rotary_emb = None,
        memory = None,
        return_memory = False
    ):
        device = x.device

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        q, k, v = (self.split_heads(t) for t in (q, k, v))

        # rotary embed

        if exists(rotary_emb):
            q = apply_rotary_emb(rotary_emb, q)
            k = apply_rotary_emb(rotary_emb, k)

        # past keys / values

        if exists(memory):
            past_k, past_v = memory
            k = cat((past_k, k), dim = -2)
            v = cat((past_v, v), dim = -2)

        # attention

        sim = einsum(q, k, 'b h i d, b h j d -> b h i j') * self.scale

        i, j = shape(sim, 'b h [i j]')
        causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)

        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        # merge and combine

        out = self.merge_heads(out)
        out = self.to_out(out)

        if not return_memory:
            return out

        return out, (k, v)

# feedforward

class FeedForward(Module):
    def __init__(
        self,
        dim,
        expansion = 4.
    ):
        super().__init__()
        dim_inner = int(dim * expansion * 2 / 3)

        self.norm = RMSNorm(dim)
        self.proj_in = Linear(dim, dim_inner * 2)
        self.proj_out = Linear(dim_inner, dim)

    def forward(self, x):
        x = self.norm(x)
        x, gates = self.proj_in(x).chunk(2, dim = -1)

        x = x * F.gelu(gates)
        return self.proj_out(x)

# transformer

class Transformer(Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_expansion = 4.,
        pad_id = -1
    ):
        super().__init__()

        # embedding

        self.num_tokens = num_tokens
        self.pad_id = pad_id

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.rotary_emb = RotaryEmbedding(dim_head)

        # attention and feedforwards

        layers = []

        for _ in range(depth):
            attn = Attention(dim = dim, dim_head = dim_head, heads = heads)
            ff = FeedForward(dim = dim, expansion = ff_expansion)

            layers.append(ModuleList([attn, ff]))

        self.layers = ModuleList(layers)

        # unembed

        self.to_logits = nn.Sequential(
            RMSNorm(dim),
            LinearNoBias(dim, num_tokens)
        )

        self.register_buffer('dummy', tensor(0), persistent = False)

    def trainable_parameter_names(self):
        return {name for name, param in self.named_parameters() if param.requires_grad}

    @torch.no_grad()
    @temp_eval
    def generate(
        self,
        batch_size = 1,
        max_length = 256,
        prompt_ids = None,
        prepend_sos = None,
        temperature = 1.,
        filter_logits_fn = top_k,
        filter_thres = 0.9,
        sos_id = 0,
        eos_ids = (0,),
        pad_value = None,
        decode_fn = None,
        mask_out_eos_on_first_step = True,
        return_for_policy_optimization = False
    ):
        pad_value = default(pad_value, self.pad_id)
        prepend_sos = default(prepend_sos, not exists(prompt_ids))

        device = self.dummy.device

        termination_ids = tensor(eos_ids, device = device)

        # a prompt is optional, sos prepended when absent - pack leading batch dims

        if exists(prompt_ids):
            prompt_ids = torch.as_tensor(prompt_ids, dtype = torch.long, device = device)
            prompt_ids, unpack = pack_with_inverse(prompt_ids, '* n')
        else:
            assert prepend_sos, 'sos must be prepended if no prompt is given'

            prompt_ids = torch.empty((batch_size, 0), dtype = torch.long, device = device)
            unpack = identity

        # one prompt per batch, may be variable length

        _, prompt_len = shape(prompt_ids, 'b n')

        # optionally seed with a start token

        out = prompt_ids

        if prepend_sos:
            out = pad_left_at_dim(prompt_ids, 1, dim = -1, value = sos_id)
            prompt_len += 1

        ids = out

        # log probs, seeded zero width in case no steps are taken

        log_probs = [out[..., :0].float()]

        memory = None

        # sample until every sequence has terminated

        for step in range(max_length):
            is_first_step = step == 0

            logits, memory = self.forward(ids, memory = memory, return_memory = True)
            logits = logits[:, -1:]

            # policy log probs from unfiltered logits, matching replay

            policy_logits = logits

            # cannot immediately terminate on first step

            if mask_out_eos_on_first_step and is_first_step:
                termination_mask = isin(arange(self.num_tokens, device = device), termination_ids)

                logits = logits.masked_fill(termination_mask, -torch.finfo(logits.dtype).max)

            # filter and sample

            filtered_logits = filter_logits_fn(logits, filter_thres)
            sample = gumbel_sample(filtered_logits, temperature = temperature)

            ids = sample
            out = cat((out, sample), dim = -1)

            # get log probs

            if return_for_policy_optimization:
                log_prob = batched_index_select(policy_logits.log_softmax(dim = -1), sample, dim = -1)

                log_probs.append(log_prob)

            # exit if all terminated - generated region only, as sos / eos id is shared

            if isin(out[:, prompt_len:], termination_ids).any(dim = -1).all():
                break

        # mask first terminator and after, generated region only

        generated = out[:, prompt_len:]

        terminated = isin(generated, termination_ids).cumsum(dim = -1) > 0

        generated.masked_fill_(terminated, pad_value)

        # generated lengths, excluding the terminator

        gen_lens = (~terminated).sum(dim = -1)

        seq_mask = pad_left_at_dim(lens_to_mask(gen_lens, max_len = size(out, '... [n]') - prompt_len), prompt_len)

        # restore leading batch dims

        out = unpack(out)
        seq_mask = unpack(seq_mask)

        program_ids = out

        # remove the prompt

        out = out[:, prompt_len:]

        # decode

        if exists(decode_fn):
            out = [decode_fn(ids) for ids in out.tolist()]

        if not return_for_policy_optimization:
            return out

        # old log probs, aligned with the full replayed sequence - pad, then unpack

        old_log_probs = unpack(pad_left_at_dim(cat(log_probs, dim = -1), prompt_len))

        policy_opt_return = SimpleNamespace(
            prompt_len = prompt_len,
            decoded_ids = program_ids,
            old_log_probs = old_log_probs,
            seq_mask = seq_mask,
        )

        return out, policy_opt_return

    def forward_with_jvp(
        self,
        ids,
        tangent,
        detach_seq_loss_tangent = True
    ):

        # only gradient requiring parameters participate
        # params always detached, as only the loss tangent is needed, the learner takes its own step

        trainable_names = self.trainable_parameter_names()

        params = tree_map_detach({
            name: param
            for name, param in self.named_parameters()
            if name in trainable_names
        })

        tangent = {name: t for name, t in tangent.items() if name in trainable_names}

        # per sequence loss for the tangent (generator reward), accounting for padding

        def functional_forward(p):
            loss, loss_mask = functional_call(self, p, (ids,), dict(return_loss = True, reduce_loss = False))

            return masked_mean(loss, loss_mask, dim = -1)

        # jvp

        seq_loss, seq_loss_tangent = jvp(functional_forward, (params,), (tangent,))

        # just detach by default, as it is used as rewards downstream

        if detach_seq_loss_tangent:
            seq_loss_tangent = seq_loss_tangent.detach()

        return seq_loss, seq_loss_tangent

    def forward(
        self,
        ids,
        memory = None,
        return_loss = False,
        return_memory = False,
        reduce_loss = False,
        loss_weights = None
    ):
        has_memory = exists(memory)

        assert not (has_memory and return_loss), 'return loss cannot be turned on when a memory is passed in'
        assert not (exists(loss_weights) and not reduce_loss), 'loss weights are applied after the sequence reduction'

        # memory is (tokens seen, [(k, v), ...]) - tokens seen is shared across layers

        if not exists(memory):
            memory = (0, None)

        tokens_seen, layer_memories = memory

        pad_id, device = self.pad_id, ids.device

        if return_loss:
            ids, labels = ids[:, :-1], ids[:, 1:]

        ids = ids.masked_fill(ids == pad_id, 0)

        # tokens

        tokens = self.token_emb(ids)

        seq_len = size(tokens, 'b [n] d')

        # rotary positions for the entire sequence - all tokens seen plus new, apply_rotary_emb takes the tail

        pos = arange(tokens_seen + seq_len, device = device)
        rotary_emb = self.rotary_emb(pos)

        # attention layers, gathering next memories

        next_memories = []

        for ind, (attn, ff) in enumerate(self.layers):
            layer_memory = layer_memories[ind] if has_memory else None

            tokens, next_memory = attn(tokens, rotary_emb = rotary_emb, memory = layer_memory, return_memory = True)
            next_memories.append(next_memory)

            tokens = ff(tokens) + tokens

        # logits

        logits = self.to_logits(tokens)

        # maybe early return logits and memories

        if not return_loss:
            if not return_memory:
                return logits

            return logits, (tokens_seen + seq_len, next_memories)

        loss = F.cross_entropy(
            rearrange(logits, 'b n v -> b v n'),
            labels,
            ignore_index = pad_id,
            reduction = 'none'
        )

        loss_mask = labels != pad_id

        # unreduced token loss and mask, for the jvp tangent

        if not reduce_loss:
            return loss, loss_mask

        # sequence mean over content tokens

        seq_loss = masked_mean(loss, loss_mask, dim = -1)

        # per sequence weights, expected to already be normalized by the caller

        if exists(loss_weights):
            return (seq_loss * loss_weights).sum()

        return seq_loss.mean()

# default learner tokenizer

def char_encode(strings, add_zero_sos_eos_id = True):
    offset = int(add_zero_sos_eos_id)
    zero_sos_eos = [0] * offset

    return pad_sequence(
        [tensor(zero_sos_eos + [ord(c) + offset for c in string] + zero_sos_eos, dtype = torch.long) for string in strings],
        value = -1
    )

def char_decode(ids, add_zero_sos_eos_id = True):
    ids = ids.tolist() if is_tensor(ids) else ids

    offset = int(add_zero_sos_eos_id)
    special_ids = {0, -1} if add_zero_sos_eos_id else {-1}

    return ''.join(chr(i - offset) for i in ids if i not in special_ids)

# default learner preconditioning - the diagonal AdamW step operator

def adam_preconditioning(param, state, param_group):
    lr, eps, (_, beta2) = pick(param_group, ('lr', 'eps', 'betas'))
    exp_avg_sq, step = pick(state, ('exp_avg_sq', 'step'))

    step = step.item() if is_tensor(step) else step

    v_hat = exp_avg_sq / (1 - beta2 ** step) if step > 0 else exp_avg_sq

    return (lr / (v_hat.sqrt() + eps)).detach()

PRECONDITIONING_FNS = {
    Adam: adam_preconditioning,
    AdamW: adam_preconditioning
}

def register_preconditioning(optimizer_type, preconditioning_fn):
    PRECONDITIONING_FNS[optimizer_type] = preconditioning_fn

# learner reference - tracks past learner parameters for the reward tangent

class LearnerReference(Module):
    def update(self, learner, step):
        # ingest the learner parameters after a step
        raise NotImplementedError

    def reference_params(self, learner, step):
        # reference parameters the difference is derived against
        raise NotImplementedError

    def difference(self, learner, step):
        # reference minus current learner parameters

        reference_params = self.reference_params(learner, step)

        return {
            name: (reference_params[name].to(param) - param).detach()
            for name, param in learner.named_parameters()
            if name in reference_params and param.requires_grad
        }

# learner reference - checkpoint lookback

def default_lookback_epoch_fn(epoch):
    return epoch // 2

class CheckpointReference(LearnerReference):
    def __init__(
        self,
        folder = 'learner_checkpoints',
        lookback_epoch_fn = None,
        strict = True
    ):
        super().__init__()

        self.folder = Path(folder)
        self.folder.mkdir(parents = True, exist_ok = True)

        self.lookback_epoch_fn = default(lookback_epoch_fn, default_lookback_epoch_fn)
        self.strict = strict

    def checkpoint_path(self, epoch):
        return self.folder / f'learner.{int(epoch)}.pt'

    def nearest_checkpoint_epoch(self, epoch):
        checkpoint_epochs = sorted(int(path.stem.split('.')[-1]) for path in self.folder.glob('learner.*.pt'))
        assert len(checkpoint_epochs) > 0, f'no learner checkpoints found in {self.folder}'
        return min(checkpoint_epochs, key = lambda e: abs(e - epoch))

    @torch.no_grad()
    def update(self, learner, step):
        torch.save(learner.state_dict(), self.checkpoint_path(step))

    def reference_params(self, learner, step):
        # learner parameters at the lookback epoch

        lookback = self.lookback_epoch_fn(step)

        if not self.strict:
            lookback = self.nearest_checkpoint_epoch(lookback)

        path = self.checkpoint_path(lookback)
        assert path.exists(), f'no learner checkpoint found at {path}'

        return torch.load(path, map_location = 'cpu', weights_only = True)

# learner reference - exponential moving average

class EMAReference(LearnerReference):
    def __init__(self, decay = 0.99):
        super().__init__()

        self.decay = decay
        self.ema = None

    @torch.no_grad()
    def update(self, learner, step):
        if not exists(self.ema):
            self.ema = EMA(
                learner,
                beta = self.decay,
                update_after_step = 0,
                update_every = 1,
                include_online_model = False
            )

        self.ema.update()

    def reference_params(self, learner, step):
        assert exists(self.ema), 'ema reference has not been initialized'

        return dict(self.ema.ema_model.named_parameters())

# classes

# quality diversity archive - elite programs per structural niche

def program_length_descriptor(program, output = None, execution_info = None):
    return len(program)

class QualityDiversityArchive(Module):
    def __init__(
        self,
        descriptor_fns = None,
        *,
        max_programs_per_niche = 8,
        reward_decay = 0.97
    ):
        super().__init__()

        # each descriptor fn -> one niche coordinate

        self.descriptor_fns = default(descriptor_fns, (program_length_descriptor,))

        self.max_programs_per_niche = max_programs_per_niche
        self.reward_decay = reward_decay

        # niche -> entries of (reward, program, output)

        self.archive = dict()

    def derive_descriptors(self, program, output = None, execution_info = None):
        # program -> niche key
        raise NotImplementedError

    def add(self, program, output, reward, execution_info = None):
        # keep top per niche
        raise NotImplementedError

    def remove(self, program):
        raise NotImplementedError

    def advance_age(self):
        # decay rewards so stale elites get displaced
        raise NotImplementedError

    def mutate(self, program):
        # single-token substitution, insertion, or deletion
        raise NotImplementedError

# self play

class SelfPlay(Module):
    def __init__(
        self,
        generator: Module,
        learner: Module,
        executor: Executor,
        accelerator = None,
        cpu = False,
        learned_tokenizer_encode = None,
        learner_optimizer = None,
        learner_preconditioning_fn = None,
        learner_lr = 3e-4,
        learner_weight_decay = 0.01,
        learner_max_grad_norm = 1.,
        generator_optimizer = None,
        generator_lr = 3e-4,
        generator_weight_decay = 0.01,
        generator_max_grad_norm = 1.,
        expert_iter_loss_weight = 1.0,
        learner_reference = None
    ):
        super().__init__()

        # accelerator, handles device placement (mps on apple silicon) and distributed training

        self.accelerator = default(accelerator, Accelerator(cpu = cpu))

        # generator and executor must share the same vocabulary size

        if isinstance(generator, Transformer):
            assert generator.num_tokens == executor.num_tokens, 'generator and executor must share the same vocabulary size'

        self.generator = generator
        self.learner = learner
        self.executor = executor

        # default learner tokenizer

        self.learned_tokenizer_encode = default(learned_tokenizer_encode, char_encode)

        # default learner optimizer

        if not exists(learner_optimizer):
            learner_optimizer = AdamW(learner.parameters(), lr = learner_lr, weight_decay = learner_weight_decay)

        self.learner_optimizer = learner_optimizer

        # generator optimizer

        if not exists(generator_optimizer):
            generator_optimizer = AdamW(generator.parameters(), lr = generator_lr, weight_decay = generator_weight_decay)

        self.generator_optimizer = generator_optimizer

        self.learner_max_grad_norm = learner_max_grad_norm
        self.generator_max_grad_norm = generator_max_grad_norm

        # weight of the reward weighted sft term - eq. 5 of the paper

        self.expert_iter_loss_weight = expert_iter_loss_weight

        # default learner preconditioning derived from the optimizer

        self.learner_preconditioning_fn = default(learner_preconditioning_fn, PRECONDITIONING_FNS.get(type(learner_optimizer)))

        # generator prior for the kl, uniform by default

        self.generator_prior = None

        # hand models and optimizers over to accelerate for placement and wrapping

        (
            self.generator,
            self.learner,
            self.generator_optimizer,
            self.learner_optimizer
        ) = self.accelerator.prepare(
            generator,
            learner,
            generator_optimizer,
            learner_optimizer
        )

        # reference learner parameters the generator reward is derived against, checkpoint lookback by default

        if not exists(learner_reference):
            learner_reference = CheckpointReference()

        self.learner_reference = learner_reference

        # persistent step counter shared with the learner reference

        self.register_buffer('epoch', tensor(0, dtype = torch.long))

        # seed the reference with the initial learner parameters

        self.learner_reference.update(self.unwrapped_learner, self.epoch.item())

    @property
    def device(self):
        return self.accelerator.device

    @property
    def unwrapped_generator(self):
        return self.accelerator.unwrap_model(self.generator)

    @property
    def unwrapped_learner(self):
        return self.accelerator.unwrap_model(self.learner)

    def parameter_difference(self):
        # reference minus current learner parameters

        return self.learner_reference.difference(self.unwrapped_learner, self.epoch.item())

    @property
    def learner_preconditioning(self):
        preconditioning_fn = self.learner_preconditioning_fn
        assert exists(preconditioning_fn), 'no learner preconditioning fn could be derived from the optimizer'

        param_groups = {
            id(param): param_group
            for param_group in self.learner_optimizer.param_groups
            for param in param_group['params']
        }

        optimizer_state = self.learner_optimizer.state

        preconditioning = dict()

        for name, param in self.unwrapped_learner.named_parameters():
            if not param.requires_grad:
                continue

            param_group = param_groups.get(id(param), dict())
            state = optimizer_state.get(param, dict())
            preconditioning[name] = preconditioning_fn(param, state, param_group)

        return preconditioning

    def preconditioned_parameter_difference(self):
        # preconditioned reference difference, i.e. the generator reward

        difference = self.parameter_difference()
        preconditioning = self.learner_preconditioning

        return {name: preconditioning[name] * diff for name, diff in difference.items()}

    @torch.no_grad()
    def advance_generator_prior(self):
        # snapshot the current generator as the new prior

        self.generator_prior = deepcopy(self.unwrapped_generator)
        self.generator_prior.eval()

    @torch.no_grad()
    def prior_log_probs(
        self,
        replay_ids,
        seq_mask,
        prior_generator = None
    ):
        # prior generator - advanced past the fixed uniform prior

        if exists(prior_generator):
            logits = prior_generator(replay_ids, return_loss = False)
            log_probs = batched_index_select(logits.log_softmax(dim = -1), replay_ids, dim = -1)

            return masked_sum(log_probs, seq_mask, dim = -1)

        # uniform prior over programs - shorter programs are more likely

        generator = self.unwrapped_generator

        lens = seq_mask.sum(dim = -1) + (replay_ids == generator.pad_id).any(dim = -1)

        return -lens * math.log(generator.num_tokens)

    @move_inputs_to_module_device
    def grpo_loss(
        self,
        rewards,         # (b)
        *,
        old_log_probs,   # (b n)
        replay_ids,      # (b n)
        prompt_len = 1,  # 1 for start token
        log_ratio_clamp = (-20., 20.),
        length_normalize = False,
        length_normalize_kl = False,
        seq_mask = None,
        kl_loss_weight = 1.,
        prior_generator = None,
    ):
        generator = self.unwrapped_generator

        if not exists(seq_mask):
            seq_mask = replay_ids != generator.pad_id
            seq_mask[:, :prompt_len] = False

        # replay log probs

        replay_logits = generator(replay_ids, return_loss = False)
        replay_log_probs = replay_logits.log_softmax(dim = -1)

        token_log_probs = batched_index_select(replay_log_probs, replay_ids, dim = -1)

        log_probs = masked_sum(token_log_probs, seq_mask, dim = -1)
        old_log_probs = masked_sum(old_log_probs, seq_mask, dim = -1)

        # advantage - normalized rewards minus the kl to the generator prior

        prior_log_probs = self.prior_log_probs(replay_ids, seq_mask, prior_generator = prior_generator)

        seq_lens = seq_mask.sum(dim = -1).clamp_min(1.)

        normed_rewards = z_score(rewards)
        kl_to_prior = (log_probs - prior_log_probs).detach()

        if length_normalize_kl:
            kl_to_prior = kl_to_prior / seq_lens

        advantages = normed_rewards - kl_loss_weight * kl_to_prior

        # sequence importance ratio

        ratio_mult = seq_lens ** -1. if length_normalize else 1.

        ratio = (log_probs - old_log_probs).mul(ratio_mult).clamp(*log_ratio_clamp).exp()

        # grpo loss

        grpo_loss = -(ratio * advantages).mean()

        return grpo_loss, (normed_rewards, kl_to_prior), log_probs

    def expert_iter_loss(
        self,
        rewards,    # (b)
        log_probs   # (b)
    ):
        # reward weighted sft on the programs - eq. 5

        weights = rewards_to_loss_weights(rewards)

        return -(weights * log_probs).sum()

    def forward(
        self,
        batch_size = 16,
        max_length = 256,
        temperature = 1.,
        filter_thres = 0.9,
        verbose = True,
        epochs = 1,
        decode_fn = None
    ):
        assert batch_size > 1, 'batch size must be greater than 1 for grpo'
        assert epochs >= 1

        # default decode fn is the executor's

        decode_fn = default(decode_fn, self.executor.decode)

        # accumulants

        losses = []
        loss_tangents = []

        for _ in range(epochs):
            # generate a batch of decoded programs, listening for eos

            eos_ids = tuple(eid for eid in (self.executor.sos_eos_id, self.executor.halt_id) if exists(eid))

            generator, learner = self.unwrapped_generator, self.unwrapped_learner

            programs, program_generate_intermediates = generator.generate(
                batch_size = batch_size,
                max_length = max_length,
                temperature = temperature,
                filter_thres = filter_thres,
                sos_id = self.executor.sos_eos_id,
                eos_ids = eos_ids,
                decode_fn = decode_fn,
                return_for_policy_optimization = True
            )

            # execute the programs

            outputs = [self.executor(program) for program in programs]

            if verbose:
                for program, output in zip(programs, outputs):
                    print(f'{program!r} -> {output!r}')

            # encode executor outputs for the learner

            ids = self.learned_tokenizer_encode(outputs).to(self.device)

            # learner step on next token prediction

            loss = learner(ids, return_loss = True, reduce_loss = True)
            self.accelerator.backward(loss)

            if exists(self.learner_max_grad_norm):
                self.accelerator.clip_grad_norm_(learner.parameters(), self.learner_max_grad_norm)

            self.learner_optimizer.step()
            self.learner_optimizer.zero_grad()

            # advance step, fold the learner step into the reference

            self.epoch.add_(1)
            self.learner_reference.update(learner, self.epoch.item())

            # preconditioned difference to the reference, used as the generator reward

            tangent = self.preconditioned_parameter_difference()

            _, loss_tangent = learner.forward_with_jvp(ids, tangent)

            # generator reward - alignment with the learner's preconditioned parameter movement

            rewards = loss_tangent.abs()

            program_ids = program_generate_intermediates.decoded_ids
            old_log_probs = program_generate_intermediates.old_log_probs
            prompt_len = program_generate_intermediates.prompt_len

            rl_loss, rl_loss_breakdown, program_log_probs = self.grpo_loss(
                rewards,
                replay_ids = program_ids,
                old_log_probs = old_log_probs,
                prompt_len = prompt_len,
                seq_mask = program_generate_intermediates.seq_mask,
                prior_generator = self.generator_prior
            )

            # expert iter - reward weighted sft on the same programs

            ei_loss = self.expert_iter_loss(rewards, program_log_probs)
            generator_loss = rl_loss + self.expert_iter_loss_weight * ei_loss

            self.accelerator.backward(generator_loss)

            if exists(self.generator_max_grad_norm):
                self.accelerator.clip_grad_norm_(generator.parameters(), self.generator_max_grad_norm)

            self.generator_optimizer.step()
            self.generator_optimizer.zero_grad()

            # accumulate

            losses.append(loss.detach())
            loss_tangents.append(loss_tangent)

        return stack(losses), stack(loss_tangents)
