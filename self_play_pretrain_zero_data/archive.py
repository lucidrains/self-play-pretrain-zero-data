from __future__ import annotations

from dataclasses import dataclass
from random import choice, choices

import torch
from torch import Tensor, arange, cat, is_tensor, randint, tensor
from torch.nn import Module

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

def substitute(ids, num_tokens, generator = None):
    if ids.numel() == 0:
        return insert(ids, num_tokens, generator = generator)

    position = randint(ids.numel(), (1,), device = ids.device, generator = generator)
    token = randint(1, num_tokens, (1,), device = ids.device, generator = generator)

    return ids.scatter(-1, position, token)

def insert(ids, num_tokens, generator = None):
    position = randint(ids.numel() + 1, (1,), device = ids.device, generator = generator)
    token = randint(1, num_tokens, (1,), device = ids.device, generator = generator)

    return cat((ids[:position], token, ids[position:]))

def delete(ids, num_tokens, generator = None):
    if ids.numel() == 0:
        return insert(ids, num_tokens, generator = generator)

    position = randint(ids.numel(), (1,), device = ids.device, generator = generator)
    keep = arange(ids.numel(), device = ids.device) != position

    return ids[keep]

# crossover - k point recombination

def crossover(ids_a, ids_b, num_points = 1, generator = None):
    def sample_bounds(ids):
        cuts = randint(ids.numel() + 1, (num_points,), device = ids.device, generator = generator).sort().values.tolist()
        return [0, *cuts, ids.numel()]

    bounds_a, bounds_b = sample_bounds(ids_a), sample_bounds(ids_b)

    segments = []

    for ind in range(num_points + 1):
        ids, bounds = (ids_a, bounds_a) if ind % 2 == 0 else (ids_b, bounds_b)
        segments.append(ids[bounds[ind]:bounds[ind + 1]])

    return cat(segments)

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
