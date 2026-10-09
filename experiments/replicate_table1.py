"""minimal replication of Table 1 - recognizable math sequences discovered during self-play

trains a generator / learner pair for the selected executor, checks fresh generator
samples every round with the appendix C family detector, compares against uniform
sampling, and streams live.json for the monitor

  python experiments/replicate_table1.py --minutes 30       # brainfuck, ema reference
  python experiments/replicate_table1.py --executor forth --minutes 30
  python experiments/replicate_table1.py --reference checkpoint   # paper lookback
  python experiments/monitor.py --live experiments/runs/default/live.json
  # ablations: --ablate-generator frozen, --ablate-reward shuffle|negate, --ablate-proposals fresh
  # warm start: --resume-generator/-learner/-archive RUN/generator.pt for a finished run
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from families import FAMILIES, expected_first_round, family_of, uniform_baseline
from report import make_plot, print_summary

from self_play_pretrain_zero_data import (
    Brainfuck,
    CheckpointReference,
    EMAReference,
    Forth,
    QualityDiversityArchive,
    SelfPlay,
    Transformer,
)
from self_play_pretrain_zero_data.executors.base import exists


class LookbackCheckpointReference(CheckpointReference):
    def __init__(self, every = 16, slack = 32, **kwargs):
        super().__init__(strict = False, **kwargs)
        self.every, self.slack = every, slack

    @torch.no_grad()
    def update(self, learner, step):
        if step % self.every == 0:
            torch.save(learner.state_dict(), self.checkpoint_path(step))

        keep_from = step // 2 - self.slack

        for path in self.folder.glob('learner.*.pt'):
            if int(path.stem.split('.')[-1]) < keep_from:
                path.unlink()

# archive descriptors, paper buckets program length at 8, 16 and 32 tokens and loop depth 0..8

def program_length_bucket(execution):
    length = len(execution.program)
    return 0 if length <= 8 else 1 if length <= 16 else 2 if length <= 32 else 3

def execution_loops_bucket(execution):
    return min(execution.loops.bit_length(), 8)

def byte_head(string, count = 16):
    return [ord(char) for char in string[:count]]

def live_snapshot(args, start, round_idx, losses, discoveries, archive, baseline_hits, recent, sample_hits, num_parameters = 0, finished = False):
    expected_round = {family: expected_first_round(baseline_hits[family], args.baseline_samples, args.detect_samples)[0] for family in FAMILIES}

    archive_rows = [dict(program = entry.program, reward = round(entry.reward, 3), age = entry.age, length = len(entry.program), loops = entry.execution_info.loops,
                         output_head = byte_head(entry.execution_info.output), family = family_of(entry.execution_info.output, args.max_output))
                    for entry in heapq.nlargest(64, archive, key = lambda entry: entry.reward)]

    return dict(
        status = 'finished' if finished else 'running',
        updated = time.time(),
        round = round_idx,
        num_parameters = num_parameters,
        elapsed_min = (time.time() - start) / 60.,
        losses = [round(loss, 4) for loss in losses],
        archive_size = len(archive),
        discoveries = discoveries,
        sample_hits = sample_hits,
        samples_checked = round_idx * args.detect_samples,
        baseline = dict(num_samples = args.baseline_samples, hits = baseline_hits, expected_round = expected_round),
        recent = recent[-8:],
        archive_rows = archive_rows,
        config = {key: value for key, value in vars(args).items() if not key.startswith('_')}
    )

def write_live(path, snapshot):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(snapshot))
    temp.replace(path)

def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)

    for name, value in dict(
        rounds = 2000, minutes = 30., batch_size = 64, detect_samples = 96, max_length = 32, max_output = 96,
        max_steps = 100_000, dim = 192, depth = 3, heads = 6, dim_head = 32, learner_lr = 3e-4, generator_lr = 3e-4,
        ema_decay = 0.99, baseline_samples = 300_000, report_every = 8, threads = min(8, os.cpu_count() or 1), seed = 0
    ).items():
        parser.add_argument(f'--{name.replace("_", "-")}', type = type(value), default = value)

    parser.add_argument('--reference', choices = ('checkpoint', 'ema'), default = 'ema')
    parser.add_argument('--executor', choices = ('brainfuck', 'forth'), default = 'brainfuck')
    parser.add_argument('--ablate-generator', choices = ('none', 'frozen'), default = 'none')
    parser.add_argument('--ablate-reward', choices = ('none', 'shuffle', 'negate'), default = 'none')
    parser.add_argument('--ablate-proposals', choices = ('none', 'fresh'), default = 'none')
    parser.add_argument('--kl-weight', type = float, default = 1.)
    parser.add_argument('--length-normalize-kl', action = 'store_true')
    parser.add_argument('--bucketed-archive', action = 'store_true')
    parser.add_argument('--device', choices = ('cpu', 'mps'), default = 'cpu')
    parser.add_argument('--resume-generator', type = str)
    parser.add_argument('--resume-learner', type = str)
    parser.add_argument('--resume-archive', type = str)
    parser.add_argument('--out', type = str, default = 'experiments/runs/default')

    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    out = Path(args.out)
    out.mkdir(parents = True, exist_ok = True)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    executor = dict(brainfuck = Brainfuck, forth = Forth)[args.executor](max_output_len = args.max_output, max_steps = args.max_steps)

    generator = Transformer(num_tokens = executor.num_tokens, dim = args.dim, depth = args.depth, dim_head = args.dim_head, heads = args.heads)
    learner = Transformer(num_tokens = 256 + 1, dim = args.dim, depth = args.depth, dim_head = args.dim_head, heads = args.heads)

    archive = QualityDiversityArchive(descriptor_fns = (program_length_bucket, execution_loops_bucket), executor = executor) if args.bucketed_archive else QualityDiversityArchive(executor = executor)

    for path, module in ((args.resume_generator, generator), (args.resume_learner, learner)):
        if exists(path):
            module.load_state_dict(torch.load(path, map_location = 'cpu', weights_only = True))

    if exists(args.resume_archive):
        archive.load(args.resume_archive)

    reference = EMAReference(decay = args.ema_decay) if args.reference == 'ema' else LookbackCheckpointReference(folder = out / 'checkpoints')
    generator_lr = 0. if args.ablate_generator == 'frozen' else args.generator_lr
    proposals = None if args.ablate_proposals == 'none' else dict(fresh = 1.)

    self_play = SelfPlay(
        generator,
        learner,
        executor,
        archive = archive,
        cpu = args.device == 'cpu',
        learner_lr = args.learner_lr,
        generator_lr = generator_lr,
        learner_reference = reference,
        proposals = proposals
    )

    if args.kl_weight != 1. or args.length_normalize_kl:
        # the unnormalized sequence KL can dominate the z-scored reward, the paper tunes this weight

        def grpo_with_kl(rewards, **kwargs):
            kwargs.update(kl_loss_weight = args.kl_weight, length_normalize_kl = args.length_normalize_kl)
            return original_grpo(rewards, **kwargs)

        original_grpo = self_play.grpo_loss
        self_play.grpo_loss = grpo_with_kl

    if args.ablate_reward != 'none':

        def ablated_reward(loss_fn):
            def loss_with_ablated_reward(rewards, *args_, **kwargs):
                rewards = rewards[torch.randperm(len(rewards), device = rewards.device)] if args.ablate_reward == 'shuffle' else -rewards
                return loss_fn(rewards, *args_, **kwargs)
            return loss_with_ablated_reward

        self_play.grpo_loss, self_play.expert_iter_loss = ablated_reward(self_play.grpo_loss), ablated_reward(self_play.expert_iter_loss)

    print(f'generator {generator.num_parameters / 1e6:.2f}M params · learner {learner.num_parameters / 1e6:.2f}M params · device {self_play.device}', flush = True)

    losses, discoveries, recent = [], {}, []
    sample_hits, baseline_hits = dict.fromkeys(FAMILIES, 0), dict.fromkeys(FAMILIES, 0)
    start, round_idx = time.time(), 0

    def save(finished = False):
        write_live(out / 'live.json', live_snapshot(args, start, round_idx, losses, discoveries, archive, baseline_hits, recent, sample_hits, num_parameters = generator.num_parameters, finished = finished))

    save()

    if args.baseline_samples > 0:
        baseline_hits = uniform_baseline(executor, args.baseline_samples, args.seed)
        save()

    print(f'uniform baseline over {args.baseline_samples} programs -> {baseline_hits}', flush = True)

    while round_idx < args.rounds and (time.time() - start) / 60. < args.minutes:
        loss = self_play(batch_size = args.batch_size, max_length = args.max_length, verbose = False, epochs = 1, decode_fn = executor.decode, filter_thres = 0.)[0].item()
        losses.append(loss)

        # fresh generator samples, checked against the appendix C family detector

        programs = self_play.unwrapped_generator.generate(
            batch_size = args.detect_samples, max_length = args.max_length, sos_id = executor.sos_eos_id,
            eos_ids = tuple(eid for eid in (executor.sos_eos_id, executor.halt_id) if exists(eid)), decode_fn = executor.decode, filter_thres = 0.
        )

        new_family = False

        for program in programs:
            info = executor.execute(program, seed = args.seed)
            family = family_of(info.output, args.max_output)

            recent.append(dict(program = program, output_head = byte_head(info.output), family = family))

            if not exists(family):
                continue

            sample_hits[family] += 1

            if family not in discoveries:
                discoveries[family] = dict(round = round_idx, program = program, output_head = byte_head(info.output, 12), steps = info.steps, loops = info.loops)
                new_family = True

            discoveries[family]['last_seen_round'] = round_idx

        recent = recent[-8:]
        round_idx += 1

        if new_family or round_idx % args.report_every == 0:
            save()
            print(f'round {round_idx} loss {loss:.3f} elapsed {(time.time() - start) / 60.:.1f}m archive {len(archive)} families {sorted(discoveries)}', flush = True)

    archive_examples = {}

    for entry in archive:
        family = family_of(entry.execution_info.output, args.max_output)

        if exists(family):
            archive_examples.setdefault(family, dict(program = entry.program, reward = entry.reward, output_head = byte_head(entry.execution_info.output)))

    final = live_snapshot(args, start, round_idx, losses, discoveries, archive, baseline_hits, recent, sample_hits, num_parameters = generator.num_parameters, finished = True)
    final.update(rounds_completed = round_idx, archive_examples = archive_examples)
    (out / 'results.json').write_text(json.dumps(final, indent = 2))
    save(True)

    torch.save(self_play.unwrapped_generator.state_dict(), out / 'generator.pt')
    torch.save(self_play.unwrapped_learner.state_dict(), out / 'learner.pt')
    archive.save(out / 'archive.pt')

    print_summary(final)
    make_plot(final, out / 'discoveries.png')
    print(f'\nwrote {out}/results.json, {out}/live.json, {out}/discoveries.png, {out}/generator.pt, {out}/learner.pt, {out}/archive.pt')

if __name__ == '__main__':
    main()
