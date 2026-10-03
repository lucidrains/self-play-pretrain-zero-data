from __future__ import annotations
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Linear, RMSNorm

from einops import einsum
from einops.layers.torch import Rearrange

from torch_einops_utils.shape import shape, size

from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

from self_play_pretrain_zero_data.executors import Executor

# constants

LinearNoBias = partial(Linear, bias = False)

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def first(seq):
    return seq[0]

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

        self.split_heads = Rearrange('b n (h d ) -> b h n d', h = heads)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')

    def forward(
        self,
        x,
        rotary_emb = None
    ):
        device = x.device

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        q, k, v = (self.split_heads(t) for t in (q, k, v))

        if exists(rotary_emb):
            q = apply_rotary_emb(rotary_emb, q)
            k = apply_rotary_emb(rotary_emb, k)

        sim = einsum(q, k, 'b h i d, b h j d -> b h i j') * self.scale

        i, j = shape(sim, 'b h [i j]')
        causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)

        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)

        out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        out = self.merge_heads(out)
        return self.to_out(out)

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
        ff_expansion = 4.
    ):
        super().__init__()

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.rotary_emb = RotaryEmbedding(dim_head)

        layers = []

        for _ in range(depth):
            attn = Attention(dim = dim, dim_head = dim_head, heads = heads)
            ff = FeedForward(dim = dim, expansion = ff_expansion)

            layers.append(ModuleList([attn, ff]))

        self.layers = ModuleList(layers)

        self.to_logits = nn.Sequential(
            RMSNorm(dim),
            LinearNoBias(dim, num_tokens)
        )

    def forward(
        self,
        ids
    ):
        device = ids.device

        tokens = self.token_emb(ids)
        pos = torch.arange(size(ids, 'b [n]'), device = device)

        rotary_emb = self.rotary_emb(pos)

        for attn, ff in self.layers:
            tokens = attn(tokens, rotary_emb = rotary_emb) + tokens
            tokens = ff(tokens) + tokens

        logits = self.to_logits(tokens)
        return logits

# classes

class SelfPlay(Module):
    def __init__(
        self,
        generator: Module,
        learner: Module,
        executor: Executor
    ):
        super().__init__()
        self.generator = generator
        self.learner = learner
        self.executor = executor

    def execute_programs(
        self,
        programs: list[str],
        **kwargs
    ) -> list[str]:

        return [self.executor(program, **kwargs) for program in programs]

    def forward(
        self,
        **kwargs
    ):
        raise NotImplementedError
