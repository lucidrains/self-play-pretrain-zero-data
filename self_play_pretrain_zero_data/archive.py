from __future__ import annotations

from dataclasses import dataclass
from random import choice, choices

import torch
from torch import Tensor, arange, cat, is_tensor, rand, randint, tensor, where
from torch.nn import Module

from torch_einops_utils import lens_to_mask, pad_left_at_dim, pad_right_at_dim, pad_sequence
from torch_einops_utils.shape import shape, size

from self_play_pretrain_zero_data.executors.base import (
    ExecutionInfo,
    Executor,
    default,
    exists
)

# descriptors

def program_length_descriptor(execution: ExecutionInfo):
    return len(execution.program)

def execution_steps_descriptor(execution: ExecutionInfo):
    return execution.steps

def execution_loops_descriptor(execution: ExecutionInfo):
    return execution.loops

# mutation operators - single token substitutions, insertions or deletions, appendix G of the paper
# batched over right padded programs - ids (b n) long, mask (b n) bool, true for a real token

def sample_token(batch_shape, num_tokens, device, generator = None):
    # id 0 is reserved for sos / eos
    return randint(1, num_tokens, batch_shape, device = device, generator = generator)

def sample_index(count, generator = None):
    # a uniform index in [0, count) per row
    return (rand(count.shape, device = count.device, generator = generator) * count).long()

def sample_position(mask, generator = None):
    # physical index of one uniformly chosen valid token per row, 0 if empty
    target = sample_index(mask.sum(dim = -1), generator = generator)
    chosen = mask.long().cumsum(dim = -1) > target[:, None]
    return chosen.int().argmax(dim = -1)

def substitute_batch(ids, mask, num_tokens, generator = None):
    # replace one valid token per row, an empty program degrades to an insertion

    b, n = shape(ids, 'b n')

    if n == 0:
        return insert_batch(ids, mask, num_tokens, generator = generator)

    position = sample_position(mask, generator = generator)[:, None]
    token = sample_token((b,), num_tokens, ids.device, generator = generator)[:, None]

    return ids.scatter(-1, position, token), mask.scatter(-1, position, True)

def insert_batch(ids, mask, num_tokens, generator = None):
    # splice one token into a uniform gap per row

    b, n = shape(ids, 'b n')
    device = ids.device

    position = sample_index(mask.sum(dim = -1) + 1, generator = generator)
    token = sample_token((b,), num_tokens, device, generator = generator)

    indices = arange(n, device = device)
    dest = indices + (indices >= position[:, None])

    ids_out = ids.new_zeros((b, n + 1))
    mask_out = mask.new_zeros((b, n + 1))

    ids_out.scatter_(1, dest, ids)
    mask_out.scatter_(1, dest, mask)

    ids_out.scatter_(1, position[:, None], token[:, None])
    mask_out.scatter_(1, position[:, None], True)

    return ids_out, mask_out

def delete_batch(ids, mask, num_tokens, generator = None):
    # drop one valid token per row, compacting left, an empty program degrades to an insertion

    b, n = shape(ids, 'b n')
    device = ids.device

    if n == 0:
        return insert_batch(ids, mask, num_tokens, generator = generator)

    lengths = mask.sum(dim = -1)
    empty = lengths == 0

    position = sample_position(mask, generator = generator)

    # pull everything after the doomed token one to the left

    indices = arange(n, device = device)
    src = (indices + (indices >= position[:, None])).clamp(max = n - 1)

    ids = ids.gather(-1, src)

    # an empty program gains a token at the front

    token = sample_token((b,), num_tokens, device, generator = generator)
    ids = where(empty[:, None] & (indices == 0), token[:, None], ids)
    mask = lens_to_mask(where(empty, 1, lengths - 1), max_len = n)

    return ids, mask

# crossover - k point recombination, segments alternate between the two parents

def crossover_batch(ids_a, mask_a, ids_b, mask_b, num_points = 1, generator = None):
    # a dead column keeps the gathers happy even when a parent is empty

    ids_a, mask_a = pad_right_at_dim(ids_a, 1), pad_right_at_dim(mask_a, 1)
    ids_b, mask_b = pad_right_at_dim(ids_b, 1), pad_right_at_dim(mask_b, 1)

    device = ids_a.device

    b, n_a = shape(ids_a, 'b n')
    _, n_b = shape(ids_b, 'b n')

    lengths_a, lengths_b = mask_a.sum(dim = -1), mask_b.sum(dim = -1)

    def sample_bounds(lengths):
        # the two endpoints plus k sorted cuts
        cuts = (rand((b, num_points), device = device, generator = generator) * (lengths + 1)[:, None]).floor().long()
        return cat((lengths.new_zeros((b, 1)), cuts.sort(dim = -1).values, lengths[:, None]), dim = -1)

    bounds_a, bounds_b = sample_bounds(lengths_a), sample_bounds(lengths_b)

    # output segments alternate between the parents - even segments come from a

    parity = arange(num_points + 1, device = device) % 2
    seg_lens = where(parity == 0, bounds_a.diff(dim = -1), bounds_b.diff(dim = -1))

    cum = seg_lens.cumsum(dim = -1)
    out_length = cum[:, -1]

    max_length = int(out_length.max())

    if max_length == 0:
        return ids_a[:, :0], mask_a[:, :0]

    position = arange(max_length, device = device)

    # the segment each output position belongs to - the first cumulative length to pass it

    seg_id = (cum[:, None, :] > position[None, :, None]).int().argmax(dim = -1)

    local = position[None, :] - pad_left_at_dim(cum, 1).gather(1, seg_id)

    from_a = (seg_id % 2) == 0
    half = seg_id // 2

    # each segment starts at the cut of its parent

    index_a = (bounds_a[:, 0::2].gather(1, half) + local).clamp(max = n_a - 1)
    index_b = (bounds_b[:, 1::2].gather(1, half) + local).clamp(max = n_b - 1)

    ids = where(from_a, ids_a.gather(-1, index_a), ids_b.gather(-1, index_b))
    mask = lens_to_mask(out_length, max_len = max_length)

    return ids, mask

# single program wrappers

def promote(ids):
    # (n) -> (1 n), all valid
    return ids[None], ids.new_ones((1, size(ids, 'n')), dtype = torch.bool)

def demote(ids, mask):
    # (1 n) -> the valid tokens
    return ids[0][mask[0]]

def substitute(ids, num_tokens, generator = None):
    return demote(*substitute_batch(*promote(ids), num_tokens, generator = generator))

def insert(ids, num_tokens, generator = None):
    return demote(*insert_batch(*promote(ids), num_tokens, generator = generator))

def delete(ids, num_tokens, generator = None):
    return demote(*delete_batch(*promote(ids), num_tokens, generator = generator))

def crossover(ids_a, ids_b, num_points = 1, generator = None):
    return demote(*crossover_batch(*promote(ids_a), *promote(ids_b), num_points = num_points, generator = generator))

BATCH_MUTATION_FNS = {
    substitute: substitute_batch,
    insert: insert_batch,
    delete: delete_batch
}

# entry

@dataclass
class ArchiveEntry:
    program: str
    reward: float
    execution_info: ExecutionInfo
    age: int = 0
    ids: Tensor | None = None

    @property
    def output(self):
        return self.execution_info.output

# quality diversity archive

class QualityDiversityArchive(Module):
    """keeps the best programs per behavior niche"""

    def __init__(
        self,
        descriptor_fns = (program_length_descriptor, execution_loops_descriptor),
        *,
        executor: Executor | None = None,
        encode_fn = None,
        decode_fn = None,
        num_tokens = None,
        max_programs_per_niche = 8,
        reward_decay = 0.97,
        max_age = None,
        mutations = None
    ):
        super().__init__()

        if callable(descriptor_fns):
            descriptor_fns = (descriptor_fns,)

        self.descriptor_fns = tuple(descriptor_fns)

        # tokenizer

        self.executor = executor
        self.encode_fn = default(encode_fn, getattr(executor, 'encode', None))
        self.decode_fn = default(decode_fn, getattr(executor, 'decode', None))
        self.num_tokens = default(num_tokens, getattr(executor, 'num_tokens', None))

        self.max_programs_per_niche = max(max_programs_per_niche, 1)
        self.reward_decay = reward_decay
        self.max_age = max_age

        # mutation fn -> sampling weight

        self.mutations = default(mutations, {substitute: 1., insert: 1., delete: 1.})

        # niche -> entries

        self.archive: dict[tuple, list[ArchiveEntry]] = dict()

    def __len__(self):
        return sum(len(entries) for entries in self.archive.values())

    def __iter__(self):
        return iter(entry for entries in self.archive.values() for entry in entries)

    def encode(self, program: str):
        assert exists(self.encode_fn), 'an encode function is needed to encode a program'
        return tensor(self.encode_fn(program), dtype = torch.long)

    def decode(self, ids):
        assert exists(self.decode_fn), 'a decode function is needed to decode program ids'
        ids = ids.tolist() if is_tensor(ids) else ids
        return self.decode_fn(ids)

    def to_ids(self, program = None):
        # parent sampled from the archive when not given

        if exists(program):
            return program if is_tensor(program) else self.encode(program)

        parent = self.sample_parent()

        return parent.ids if exists(parent.ids) else self.encode(parent.program)

    def derive_descriptors(self, execution: ExecutionInfo):
        return tuple(fn(execution) for fn in self.descriptor_fns)

    def add(self, program: str, reward: float, execution_info: ExecutionInfo | None = None):
        # only positively rewarded programs are admitted

        if reward <= 0.:
            return False

        if not exists(execution_info):
            execution_info = self.executor.execute(program) if exists(self.executor) else ExecutionInfo(program)

        entries = self.archive.setdefault(self.derive_descriptors(execution_info), [])
        ids = self.encode(program) if exists(self.encode_fn) else None
        entry = ArchiveEntry(program, reward, execution_info, ids = ids)

        # an improved resubmission replaces the elite

        for i, existing in enumerate(entries):
            if existing.program == program:
                if reward <= existing.reward:
                    return False

                entries[i] = entry
                return True

        if len(entries) < self.max_programs_per_niche:
            entries.append(entry)
            return True

        weakest_idx, weakest = min(enumerate(entries), key = lambda pair: pair[1].reward)

        if reward <= weakest.reward:
            return False

        entries[weakest_idx] = entry
        return True

    def remove(self, program: str):
        for niche in tuple(self.archive):
            entries = [entry for entry in self.archive[niche] if entry.program != program]

            if not entries:
                del self.archive[niche]
            else:
                self.archive[niche] = entries

    def advance_age(self):
        # decay rewards, drop the expired

        for niche in tuple(self.archive):
            entries = self.archive[niche]

            for entry in entries:
                entry.age += 1
                entry.reward *= self.reward_decay

            if exists(self.max_age):
                entries = [entry for entry in entries if entry.age < self.max_age]

            if not entries:
                del self.archive[niche]
            else:
                self.archive[niche] = entries

    def sample_parent(self):
        # parents are drawn uniformly across occupied niches, not across all programs

        assert len(self.archive) > 0, 'cannot sample a mutation parent from an empty archive'

        entries = choice(tuple(self.archive.values()))

        return choice(entries)

    def sample_parents(self, num_parents):
        # parents drawn uniformly across occupied niches, right padded into a batch

        assert num_parents > 0, 'at least one parent is needed'
        assert len(self.archive) > 0, 'cannot sample mutation parents from an empty archive'

        ids, lengths = pad_sequence([self.to_ids() for _ in range(num_parents)], return_lens = True)
        mask = lens_to_mask(lengths, max_len = size(ids, 'b [n]'))

        return ids, mask

    def decode_batch(self, ids, mask):
        # strip padding, then decode each row

        lengths = mask.sum(dim = -1).tolist()

        return [self.decode(row[:length]) for row, length in zip(ids.tolist(), lengths)]

    def mutate_batch(self, num_mutants = 1):
        # a batch of local mutations - one weighted operator per mutant

        assert exists(self.num_tokens), 'a vocabulary size is needed to mutate programs'

        ids, mask = self.sample_parents(num_mutants)

        mutation_fns, weights = zip(*self.mutations.items())
        chosen = tensor(choices(range(len(mutation_fns)), weights = weights, k = num_mutants), device = ids.device)

        _, n = shape(ids, 'b n')
        width = n + 1

        mutant_ids = ids.new_zeros((num_mutants, width))
        mutant_mask = mask.new_zeros((num_mutants, width))

        for op_index, mutation_fn in enumerate(mutation_fns):
            rows = chosen == op_index

            if not rows.any():
                continue

            batch_fn = BATCH_MUTATION_FNS.get(mutation_fn)
            assert exists(batch_fn), f'`{getattr(mutation_fn, "__name__", mutation_fn)}` cannot be applied to a batch'

            op_ids, op_mask = batch_fn(ids[rows], mask[rows], self.num_tokens)
            op_width = size(op_ids, '... [w]')

            mutant_ids[rows, :op_width] = op_ids
            mutant_mask[rows, :op_width] = op_mask

        return self.decode_batch(mutant_ids, mutant_mask)

    def crossover_batch(self, num_offspring = 1, num_points = 1):
        # recombine pairs of parents, k cuts each

        assert exists(self.num_tokens), 'a vocabulary size is needed to crossover programs'

        ids_a, mask_a = self.sample_parents(num_offspring)
        ids_b, mask_b = self.sample_parents(num_offspring)

        ids, mask = crossover_batch(ids_a, mask_a, ids_b, mask_b, num_points = num_points)

        return self.decode_batch(ids, mask)

    def mutate(self, program = None):
        # local mutation of a positively rewarded program

        assert exists(self.num_tokens), 'a vocabulary size is needed to mutate programs'

        mutation_fns, weights = zip(*self.mutations.items())
        mutation_fn = choices(mutation_fns, weights = weights)[0]

        return self.decode(mutation_fn(self.to_ids(program), self.num_tokens))

    def crossover(self, program_a = None, program_b = None, num_points = 1):
        # recombine two positively rewarded programs

        assert exists(self.num_tokens), 'a vocabulary size is needed to crossover programs'

        ids_a, ids_b = (self.to_ids(program) for program in (program_a, program_b))

        return self.decode(crossover(ids_a, ids_b, num_points = num_points))
