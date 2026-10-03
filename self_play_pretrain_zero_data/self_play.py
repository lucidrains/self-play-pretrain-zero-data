from __future__ import annotations
from functools import partial
from math import ceil
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, arange, cat, stack, full, tensor, is_tensor
from torch.nn import Module, ModuleList, Linear, RMSNorm
from torch.optim import Adam, AdamW
from torch.func import functional_call, jvp

from einops import einsum, rearrange
from einops.layers.torch import Rearrange

from torch_einops_utils import clamp, mask_after, masked_mean, pack_with_inverse, pad_sequence, temp_eval, tree_map_detach
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
        pad_id = -1,
        sos_eos_id = None
    ):
        super().__init__()
        assert not exists(sos_eos_id) or 0 <= sos_eos_id < num_tokens

        # embedding

        self.num_tokens = num_tokens
        self.pad_id = pad_id
        self.sos_eos_id = sos_eos_id

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

    @torch.no_grad()
    @temp_eval
    def generate(
        self,
        batch_size = 1,
        max_length = 256,
        prompt_ids = None,
        temperature = 1.,
        filter_logits_fn = top_k,
        filter_thres = 0.9,
        eos_id = None,
        pad_value = None,
        decode_fn = None,
        mask_out_eos_on_first_step = True
    ):
        eos_id = default(eos_id, self.sos_eos_id)
        pad_value = default(pad_value, eos_id)

        assert exists(eos_id), 'an eos id must be given on init or passed into generate'

        device = self.dummy.device

        # a prompt is optional, as the auto prepended sos id (0) serves as one
        # pack any leading batch dims, to be restored at the end

        if exists(prompt_ids):
            prompt_ids = torch.as_tensor(prompt_ids, dtype = torch.long, device = device)
            prompt_ids, unpack = pack_with_inverse(prompt_ids, '* n')
        else:
            prompt_ids = torch.empty((batch_size, 0), dtype = torch.long, device = device)
            unpack = identity

        # one prompt per batch, may be variable length - derive batch size and prompt length

        batch_size, prompt_len = shape(prompt_ids, 'b n')

        ids = out = prompt_ids
        memory = None

        # sample until every sequence has emitted eos

        for step in range(max_length):
            logits, memory = self.forward(ids, memory = memory, return_memory = True)
            logits = logits[:, -1:]

            # on the very first step, mask out the sos / eos id so generation cannot immediately terminate

            if mask_out_eos_on_first_step and step == 0:
                logits = logits.masked_fill(arange(self.num_tokens, device = device) == eos_id, -torch.finfo(logits.dtype).max)

            # filter and sample

            filtered_logits = filter_logits_fn(logits, filter_thres)
            sample = gumbel_sample(filtered_logits, temperature = temperature)

            ids = sample
            out = cat((out, sample), dim = -1)

            # exit if all eos

            if (out == eos_id).any(dim = -1).all():
                break

        # mask out everything after the first eos, leaving sequences that never emitted eos untouched

        out = out.masked_fill(~mask_after(out, eos_id, inclusive = True), pad_value)

        out = unpack(out[:, prompt_len:])

        # maybe decode the sampled ids into programs

        if not exists(decode_fn):
            return out

        return [decode_fn(ids) for ids in out.tolist()]

    def trainable_parameter_names(self):
        return {name for name, param in self.named_parameters() if param.requires_grad}

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

        # forward with functional call, returning token mean loss and per sequence mean loss for the tangent (generator reward), accounting for padding

        def functional_forward(p):
            loss, loss_mask = functional_call(self, p, (ids,), dict(return_loss = True, reduce_loss = False))

            token_mean_loss = masked_mean(loss, loss_mask)
            seq_mean_loss = masked_mean(loss, loss_mask, dim = -1)

            return token_mean_loss, seq_mean_loss

        # jvp

        (token_loss, _), (_, seq_loss_tangent) = jvp(functional_forward, (params,), (tangent,))

        # just detach by default, as it is used as rewards downstream

        if detach_seq_loss_tangent:
            seq_loss_tangent = seq_loss_tangent.detach()

        return token_loss, seq_loss_tangent

    def forward(
        self,
        ids,
        memory = None,
        return_loss = False,
        return_memory = False,
        reduce_loss = False
    ):
        batch = size(ids, '[b] n')
        has_memory = exists(memory)

        # auto prepend sos / eos token id (0) at position 0, only when no memory is passed in

        auto_sos = exists(self.sos_eos_id) and not has_memory

        assert not (has_memory and return_loss), 'return loss cannot be turned on when a memory is passed in'

        # memory is (tokens seen, [(k, v), ...]) - tokens seen is shared across layers

        if not exists(memory):
            memory = (0, None)

        tokens_seen, layer_memories = memory

        pad_id, device = self.pad_id, ids.device

        if return_loss:
            ids, labels = (ids, ids) if auto_sos else (ids[:, :-1], ids[:, 1:])
            ids = ids.masked_fill(ids == pad_id, default(self.sos_eos_id, 0))

        # tokens

        tokens = self.token_emb(ids)

        if auto_sos:
            sos_ids = full((batch, 1), self.sos_eos_id, dtype = torch.long, device = device)
            tokens = cat((self.token_emb(sos_ids), tokens), dim = -2)

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

        # next token loss, dropping the extra logit produced by the auto sos token

        if auto_sos:
            logits = logits[:, :-1]

        loss = F.cross_entropy(
            rearrange(logits, 'b n v -> b v n'),
            labels,
            ignore_index = pad_id,
            reduction = 'none'
        )

        loss_mask = labels != pad_id

        if reduce_loss:
            return masked_mean(loss, loss_mask)

        return loss, loss_mask

# default learner tokenizer

def char_encode(strings):
    return pad_sequence(
        [tensor([ord(c) for c in string], dtype = torch.long) for string in strings],
        value = -1
    )

# default learner lookback - the checkpoint from half of the epochs ago

def default_lookback_epoch_fn(epoch):
    return epoch // 2

# default learner preconditioning - the diagonal AdamW step operator from the paper
# P = lr / (sqrt(v_hat) + eps)

def adam_preconditioning(param, state, param_group):
    lr = param_group['lr']
    eps = param_group['eps']
    _, beta2 = param_group['betas']

    exp_avg_sq = state['exp_avg_sq']

    step = state['step']
    step = step.item() if is_tensor(step) else step

    v_hat = exp_avg_sq / (1 - beta2 ** step) if step > 0 else exp_avg_sq

    return (lr / (v_hat.sqrt() + eps)).detach()

PRECONDITIONING_FNS = {
    Adam: adam_preconditioning,
    AdamW: adam_preconditioning
}

def register_preconditioning(optimizer_type, preconditioning_fn):
    PRECONDITIONING_FNS[optimizer_type] = preconditioning_fn

# classes

class SelfPlay(Module):
    def __init__(
        self,
        generator: Module,
        learner: Module,
        executor: Executor,
        learned_tokenizer_encode = None,
        learner_optimizer = None,
        learner_preconditioning_fn = None,
        learner_lr = 3e-4,
        learner_checkpoint_folder = 'learner_checkpoints',
        learner_lookback_epoch_fn = None,
        learner_checkpoint_strict = True
    ):
        super().__init__()

        # the generator transformer must be initialized with auto sos / eos

        if isinstance(generator, Transformer):
            assert exists(generator.sos_eos_id), 'generator transformer must have the auto sos / eos id turned on'
            assert generator.num_tokens == executor.num_tokens, 'generator and executor must share the same vocabulary size'

        self.generator = generator
        self.learner = learner
        self.executor = executor

        # default learner tokenizer

        self.learned_tokenizer_encode = default(learned_tokenizer_encode, char_encode)

        # default learner optimizer

        if not exists(learner_optimizer):
            learner_optimizer = AdamW(learner.parameters(), lr = learner_lr)

        self.learner_optimizer = learner_optimizer

        # default learner preconditioning derived from the optimizer

        self.learner_preconditioning_fn = default(learner_preconditioning_fn, PRECONDITIONING_FNS.get(type(learner_optimizer)))

        # persistent epoch counter

        self.register_buffer('epoch', tensor(0, dtype = torch.long))

        # learner checkpoints, used to derive parameter differences over the lookback window

        self.learner_checkpoint_folder = Path(learner_checkpoint_folder)
        self.learner_checkpoint_folder.mkdir(parents = True, exist_ok = True)

        # lookback checkpoint selection for the generator reward

        self.learner_lookback_epoch_fn = default(learner_lookback_epoch_fn, default_lookback_epoch_fn)
        self.learner_checkpoint_strict = learner_checkpoint_strict

        self.save_checkpoint()

    def checkpoint_path(self, epoch):
        return self.learner_checkpoint_folder / f'learner.{int(epoch)}.pt'

    def nearest_checkpoint_epoch(self, epoch):
        checkpoint_epochs = sorted(int(path.stem.split('.')[-1]) for path in self.learner_checkpoint_folder.glob('learner.*.pt'))
        assert len(checkpoint_epochs) > 0, f'no learner checkpoints found in {self.learner_checkpoint_folder}'
        return min(checkpoint_epochs, key = lambda e: abs(e - epoch))

    def save_checkpoint(self, epoch = None):
        epoch = default(epoch, self.epoch.item())
        torch.save(self.learner.state_dict(), self.checkpoint_path(epoch))

    def load_checkpoint(self, epoch, strict = None):
        # strict fails on a missing checkpoint, else the nearest one is used

        if not default(strict, self.learner_checkpoint_strict):
            epoch = self.nearest_checkpoint_epoch(epoch)

        path = self.checkpoint_path(epoch)
        assert path.exists(), f'no learner checkpoint found at {path}'

        return torch.load(path, map_location = 'cpu', weights_only = True)

    def parameter_difference(self, epoch, strict = None):
        # δθ = θ_past − θ_now

        past_params = self.load_checkpoint(epoch, strict = strict)

        trainable_names = self.learner.trainable_parameter_names()

        return {
            name: (past_params[name].to(param) - param).detach()
            for name, param in self.learner.named_parameters()
            if name in trainable_names
        }

    @property
    def learner_preconditioning(self):
        preconditioning_fn = self.learner_preconditioning_fn
        assert exists(preconditioning_fn), 'no learner preconditioning fn could be derived from the optimizer'

        param_groups = {
            id(param): param_group
            for param_group in self.learner_optimizer.param_groups
            for param in param_group['params']
        }

        preconditioning = dict()

        for name, param in self.learner.named_parameters():
            if not param.requires_grad:
                continue

            param_group = param_groups.get(id(param), dict())
            state = self.learner_optimizer.state.get(param, dict())
            preconditioning[name] = preconditioning_fn(param, state, param_group)

        return preconditioning

    def preconditioned_parameter_difference(self, epoch, strict = None):
        # preconditioned lookback tangent P ⊙ δθ, i.e. the generator reward

        difference = self.parameter_difference(epoch, strict = strict)
        preconditioning = self.learner_preconditioning

        return {name: preconditioning[name] * diff for name, diff in difference.items()}

    def forward(
        self,
        batch_size = 16,
        max_length = 256,
        temperature = 1.,
        filter_thres = 0.9,
        verbose = True,
        num_epochs = 1,
        decode_fn = None
    ):
        assert batch_size > 1, 'batch size must be greater than 1 for grpo'
        assert num_epochs >= 1

        # default decode fn is the executor's

        decode_fn = default(decode_fn, self.executor.decode)

        # accumulants

        losses = []
        loss_tangents = []

        for _ in range(num_epochs):
            # generate a batch of decoded programs, listening for eos

            programs = self.generator.generate(
                batch_size = batch_size,
                max_length = max_length,
                temperature = temperature,
                filter_thres = filter_thres,
                eos_id = self.executor.sos_eos_id,
                decode_fn = decode_fn
            )

            # execute the programs

            outputs = [self.executor(program) for program in programs]

            if verbose:
                for program, output in zip(programs, outputs):
                    print(f'{program!r} -> {output!r}')

            # encode executor outputs for the learner

            ids = self.learned_tokenizer_encode(outputs)

            # learner step on next token prediction

            loss = self.learner(ids, return_loss = True, reduce_loss = True)
            loss.backward()

            self.learner_optimizer.step()
            self.learner_optimizer.zero_grad()

            # advance epoch, checkpoint the learner

            self.epoch.add_(1)
            self.save_checkpoint()

            # preconditioned lookback tangent, used as the generator reward

            tangent = self.preconditioned_parameter_difference(self.learner_lookback_epoch_fn(self.epoch.item()))

            _, loss_tangent = self.learner.forward_with_jvp(ids, tangent)

            # handle rl and reward weighted sft

            # todo - grpo / sft

            # accumulate

            losses.append(loss.detach())
            loss_tangents.append(loss_tangent)

        return stack(losses), stack(loss_tangents)
