from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, cat, tensor

from torch_einops_utils import masked_sum, pad_right_at_dim
from torch_einops_utils.shape import size

from self_play_pretrain_zero_data.executors.base import exists

# program pool - B_e = B_fresh + B_mut + B_replay, appendix G
# pool fn: (self_play, num_programs, **generation_kwargs) -> ProgramBatch | None, none sits the round out

@dataclass
class ProgramBatch:
    programs: list[str]
    decoded_ids: Tensor    # (b n) sos followed by program ids, right padded
    seq_mask: Tensor       # (b n) true for program tokens
    old_log_probs: Tensor  # (b) proposal log prob, the importance ratio denominator
    pg_mask: Tensor        # (b) rows entering the policy gradient - mutations are excluded, eq. 4
    admit: Tensor          # (b) rows written back into the archive

    def __len__(self):
        return len(self.programs)

    @classmethod
    def concat(cls, batches, pad_id = -1):
        if not batches:
            return None

        if len(batches) == 1:
            return batches[0]

        width = max(size(batch.decoded_ids, 'b [n]') for batch in batches)

        def pad(t, value):
            return pad_right_at_dim(t, width - size(t, 'b [n]'), value = value)

        return cls(
            programs = [program for batch in batches for program in batch.programs],
            decoded_ids = cat([pad(batch.decoded_ids, pad_id) for batch in batches]),
            seq_mask = cat([pad(batch.seq_mask, False) for batch in batches]),
            old_log_probs = cat([batch.old_log_probs for batch in batches]),
            pg_mask = cat([batch.pg_mask for batch in batches]),
            admit = cat([batch.admit for batch in batches])
        )

def fresh_programs(
    self_play,
    num_programs,
    max_length = 256,
    temperature = 1.,
    filter_thres = 0.9,
    decode_fn = None,
    **generation_kwargs
):
    # on-policy generator samples - global exploration, B_fresh

    executor = self_play.executor

    eos_ids = tuple(eid for eid in (executor.sos_eos_id, executor.halt_id) if exists(eid))

    programs, info = self_play.unwrapped_generator.generate(
        batch_size = num_programs,
        max_length = max_length,
        temperature = temperature,
        filter_thres = filter_thres,
        sos_id = executor.sos_eos_id,
        eos_ids = eos_ids,
        decode_fn = decode_fn,
        return_for_policy_optimization = True,
        **generation_kwargs
    )

    seq_mask = info.seq_mask

    ones = seq_mask.new_ones(num_programs, dtype = torch.bool)

    return ProgramBatch(
        programs = programs,
        decoded_ids = info.decoded_ids,
        seq_mask = seq_mask,
        old_log_probs = masked_sum(info.old_log_probs, seq_mask, dim = -1),
        pg_mask = ones,
        admit = ones
    )

def mutation_programs(self_play, num_programs, **generation_kwargs):
    # local mutations of archived elites - refinement, B_mut
    # no proposal distribution, so expert iteration only, but admitted back as new parents

    archive = self_play.archive
    assert exists(archive), 'an archive is needed to mutate programs'

    if len(archive) == 0:
        return

    return self_play.program_batch(archive.mutate_batch(num_programs), pg = False, admit = True)

def crossover_programs(self_play, num_programs, **generation_kwargs):
    # k point recombination of archived pairs, treated like mutations

    archive = self_play.archive
    assert exists(archive), 'an archive is needed to crossover programs'

    if len(archive) == 0:
        return

    return self_play.program_batch(archive.crossover_batch(num_programs), pg = False, admit = True)

def replay_programs(self_play, num_programs, **generation_kwargs):
    # archived programs replayed - preserving earlier discoveries, B_replay
    # the stored sampling log prob seeds the off-policy ratio

    archive = self_play.archive
    assert exists(archive), 'an archive is needed to replay programs'

    if len(archive) == 0:
        return

    entries = archive.sample_entries(num_programs)
    log_probs = [entry.log_prob for entry in entries]

    return self_play.program_batch(
        [entry.program for entry in entries],
        old_log_probs = tensor(log_probs, dtype = torch.float, device = self_play.device) if all(map(exists, log_probs)) else None,
        admit = False
    )

# built-in pools by name, mix with custom callables freely

POOL_FNS = dict(
    fresh = fresh_programs,
    mutation = mutation_programs,
    crossover = crossover_programs,
    replay = replay_programs
)
