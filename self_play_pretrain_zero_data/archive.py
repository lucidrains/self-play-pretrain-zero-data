from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from random import choice, randrange

from torch.nn import Module

from self_play_pretrain_zero_data.executors.base import (
    ExecutionInfo,
    Executor,
    default,
    exists
)

# descriptors - execution record to one niche coordinate

def program_length_descriptor(execution: ExecutionInfo | str):
    return len(execution if isinstance(execution, str) else execution.program)

def execution_steps_descriptor(execution: ExecutionInfo):
    return execution.steps

def execution_loops_descriptor(execution: ExecutionInfo):
    return execution.loops

# entry

@dataclass
class ArchiveEntry:
    program: str
    reward: float
    execution_info: ExecutionInfo
    age: int = 0

    @property
    def output(self):
        return self.execution_info.output

# quality diversity archive - elite programs per structural niche

class QualityDiversityArchive(Module):
    """keeps the best programs per behavior niche"""

    def __init__(
        self,
        descriptor_fns = (program_length_descriptor, execution_loops_descriptor),
        *,
        executor: Executor | None = None,
        max_programs_per_niche = 8,
        reward_decay = 0.97,
        max_age = None
    ):
        super().__init__()

        if callable(descriptor_fns):
            descriptor_fns = (descriptor_fns,)
        else:
            descriptor_fns = default(descriptor_fns, (program_length_descriptor, execution_loops_descriptor))

        self.descriptor_fns = tuple(descriptor_fns)
        self.executor = executor

        self.max_programs_per_niche = max(max_programs_per_niche, 1)
        self.reward_decay = reward_decay
        self.max_age = max_age

        # niche -> entries

        self.archive: dict[tuple, list[ArchiveEntry]] = dict()

    def __len__(self):
        return sum(len(entries) for entries in self.archive.values())

    def __iter__(self):
        return iter(entry for entries in self.archive.values() for entry in entries)

    def derive_descriptors(self, execution: ExecutionInfo | str):
        if isinstance(execution, str):
            execution = self.executor.execute(execution) if exists(self.executor) else ExecutionInfo(execution)

        return tuple(fn(execution) for fn in self.descriptor_fns)

    def add(
        self,
        program: str,
        output: str | float | ExecutionInfo | None = None,
        reward: float | ExecutionInfo | None = None,
        execution_info: ExecutionInfo | None = None
    ):
        if isinstance(output, ExecutionInfo):
            execution_info, output = output, None

        if isinstance(reward, ExecutionInfo):
            execution_info, reward = reward, None

        if isinstance(output, (int, float)) and not exists(reward):
            reward, output = float(output), None

        assert exists(reward), 'reward must be provided'

        if not exists(execution_info):
            if exists(self.executor):
                execution_info = self.executor.execute(program)
            else:
                execution_info = ExecutionInfo(program, output = default(output, ''))

        niche = self.derive_descriptors(execution_info)
        entries = self.archive.setdefault(niche, [])

        entry = ArchiveEntry(program, reward, execution_info)

        # an improved resubmission replaces the same program's elite

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
        # decay so stale elites get displaced, drop the expired

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

    def mutate(self, program: str):
        # single token substitution, insertion or deletion

        assert exists(self.executor), 'an executor is needed to mutate programs'

        ids = self.executor.encode(program)
        op = choice(('substitute', 'insert', 'delete')) if ids else 'insert'
        random_token = partial(randrange, 1, self.executor.num_tokens)

        if op == 'delete':
            del ids[randrange(len(ids))]
        elif op == 'insert':
            ids.insert(randrange(len(ids) + 1), random_token())
        else:
            ids[randrange(len(ids))] = random_token()

        return self.executor.decode(ids)
