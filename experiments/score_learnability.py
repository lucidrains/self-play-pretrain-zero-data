"""score the learnability of discovered programs over time

part A - replay each program's outputs against the run's learner checkpoints,
giving per-program loss over self-play rounds (when was it at the frontier)

part B - paper figure 3 style epiplexity - train a fresh learner from scratch on a
single program's outputs and measure the held-out loss trajectory and excess loss

usage: python experiments/score_learnability.py --results experiments/runs/table1b/results.json
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import torch
from fire import Fire
from torch.optim import AdamW
from torch_einops_utils import masked_mean

from self_play_pretrain_zero_data import Brainfuck, Transformer
from self_play_pretrain_zero_data.executors.base import default, exists
from self_play_pretrain_zero_data.self_play import char_encode

# reference programs from the paper, plus trivial and unlearnable controls

def program_set(results, executor):
    programs = {}

    for family, record in results.get('discoveries', {}).items():
        programs[f'discovered/{family}'] = record['program']

    for family, record in results.get('archive_examples', {}).items():
        programs.setdefault(f'archive/{family}', record['program'])

    programs['table/arithmetic'] = '+[.++]'
    programs['table/fibonacci'] = ',[[.C>.C>]'
    programs['table/geometric'] = '+[.L>]'
    programs['trivial/constant'] = '+[.]'
    programs['unlearnable/random-bytes'] = ',.' * 56

    return programs

def execute_outputs(executor, program, seeds):
    return [executor.execute(program, seed = seed).output for seed in seeds]

def checkpoint_rounds(checkpoint_dir):
    return sorted(int(path.stem.split('.')[-1]) for path in Path(checkpoint_dir).glob('learner.*.pt'))

@torch.no_grad()
def checkpoint_loss(model, ids, batch = 32):
    model.eval()

    losses = [masked_mean(loss, loss_mask, dim = -1) for loss, loss_mask in (model(chunk, return_loss = True) for chunk in ids.split(batch))]

    return torch.cat(losses).mean().item() / math.log(2)

# part A - per program loss across self-play learner checkpoints

def self_play_curves(results, executor, checkpoint_dir, num_seeds = 4):
    config = results['config']

    model = Transformer(
        num_tokens = 256 + 1,
        dim = config['dim'],
        depth = config['depth'],
        dim_head = config['dim_head'],
        heads = config['heads']
    ).eval()

    programs = program_set(results, executor)

    # fixed outputs per program, seeds kept away from the detection seed

    outputs = {name: execute_outputs(executor, program, range(1000, 1000 + num_seeds)) for name, program in programs.items()}
    ids = {name: char_encode(output) for name, output in outputs.items()}

    curves, rounds = {name: [] for name in programs}, []

    for round in checkpoint_rounds(checkpoint_dir):
        path = Path(checkpoint_dir) / f'learner.{round}.pt'

        if not path.exists():
            continue

        model.load_state_dict(torch.load(path, map_location = 'cpu', weights_only = True))
        rounds.append(round)

        for name, name_ids in ids.items():
            curves[name].append(checkpoint_loss(model, name_ids))

        print(f'  checkpoint {round} done')

    return rounds, curves

# part B - fresh learner epiplexity per program

def fresh_learner_curve(
    executor,
    program,
    config,
    *,
    train_samples = 64,
    eval_samples = 16,
    steps = 300,
    batch = 16,
    eval_every = 10,
    lr = 3e-4,
    seed = 0,
    device = 'cpu'
):
    torch.manual_seed(seed)
    random.seed(seed)

    learner = Transformer(
        num_tokens = 256 + 1,
        dim = config['dim'],
        depth = config['depth'],
        dim_head = config['dim_head'],
        heads = config['heads']
    ).to(device)

    optimizer = AdamW(learner.parameters(), lr = lr, weight_decay = 0.01)

    train_ids = char_encode(execute_outputs(executor, program, range(train_samples))).to(device)
    eval_ids = char_encode(execute_outputs(executor, program, range(10_000, 10_000 + eval_samples))).to(device)

    eval_losses, steps_logged = [], []

    for step in range(steps + 1):
        if step % eval_every == 0:
            eval_losses.append(checkpoint_loss(learner, eval_ids))
            steps_logged.append(step)

        if step == steps:
            break

        learner.train()

        index = torch.randperm(len(train_ids))[:batch]
        loss = learner(train_ids[index], return_loss = True, reduce_loss = True)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(learner.parameters(), 1.)
        optimizer.step()

    floor = sum(eval_losses[-3:]) / 3

    # epiplexity - excess loss above the converged floor, in bits / byte * steps

    excess = sum((loss - floor) * eval_every for loss in eval_losses if loss > floor)

    return dict(
        steps = steps_logged,
        eval_bits_per_byte = eval_losses,
        floor_bits_per_byte = floor,
        excess_loss_steps = excess
    )

def score(
    results,
    checkpoint_dir = None,
    out = None,
    part = 'ab',
    steps = 300,
    threads = 3
):
    torch.set_num_threads(threads)

    results_path = Path(results)
    results = json.loads(results_path.read_text())
    config = results['config']

    executor = Brainfuck(max_output_len = config['max_output'])

    out = Path(default(out, results_path.parent / 'learnability'))
    out.mkdir(parents = True, exist_ok = True)

    checkpoint_dir = default(checkpoint_dir, str(results_path.parent / 'checkpoints'))

    # merge into any existing scores so parts can be recomputed independently

    saved = dict(config = config, discoveries = results.get('discoveries', {}), part_a = None, part_b = None)

    if (out / 'learnability.json').exists():
        existing = json.loads((out / 'learnability.json').read_text())
        saved.update({key: existing.get(key) for key in ('part_a', 'part_b')})

    if 'a' in part:
        print('part A - per program loss across self-play checkpoints')
        rounds, curves = self_play_curves(results, executor, checkpoint_dir)
        saved['part_a'] = dict(rounds = rounds, curves = curves)
        print(f'  {len(rounds)} checkpoints, {len(curves)} programs')

    if 'b' in part:
        print('part B - fresh learner epiplexity per program')

        part_b = saved['part_b'] or {}
        start = time.time()

        for name, program in program_set(results, executor).items():
            part_b[name] = fresh_learner_curve(executor, program, config, steps = steps)
            print(f'  {name} floor {part_b[name]["floor_bits_per_byte"]:.3f} bits/byte, excess {part_b[name]["excess_loss_steps"]:.1f} ({time.time() - start:.0f}s)')

        saved['part_b'] = part_b

    (out / 'learnability.json').write_text(json.dumps(saved, indent = 2))

    make_plot(saved, out / 'learnability.png')
    print(f'wrote {out}/learnability.json, {out}/learnability.png')

def make_plot(saved, path):
    try:
        import matplotlib
    except ImportError:
        print('matplotlib is not installed, skipping the figure')
        return

    matplotlib.use('Agg')

    import matplotlib.pyplot as plt

    config = saved['config']
    fig, axes = plt.subplots(1, 2, figsize = (13, 5))

    colors = plt.get_cmap('tab10').colors
    labels = list(saved['part_b'].keys()) if saved['part_b'] else list(saved['part_a']['curves'].keys())
    col = {name: colors[i % len(colors)] for i, name in enumerate(labels)}

    if exists(saved['part_a']):
        ax, rounds = axes[0], saved['part_a']['rounds']

        for name, curve in saved['part_a']['curves'].items():
            ax.plot(rounds, curve, label = name, color = col.get(name, 'gray'))

        for name in labels:
            if not name.startswith('discovered/'):
                continue

            record = saved.get('discoveries', {}).get(name.split('/')[1])

            if exists(record) and rounds[0] <= record['round'] <= rounds[-1]:
                ax.axvline(record['round'], color = col.get(name, 'gray'), linestyle = ':', alpha = 0.6)

        ax.set_xlabel('self-play round')
        ax.set_ylabel('learner loss (bits / byte)')
        ax.set_title('replayed against self-play learner checkpoints', fontsize = 10)
        ax.grid(alpha = 0.3)
        ax.legend(fontsize = 7, loc = 'best')

    if exists(saved['part_b']):
        ax = axes[1]

        for name, curve in saved['part_b'].items():
            ax.plot(curve['steps'], curve['eval_bits_per_byte'], label = name, color = col.get(name, 'gray'))

        ax.set_xlabel('fresh learner step')
        ax.set_ylabel('held-out loss (bits / byte)')
        ax.set_title(f'fresh learner trained on each program alone\n({config["dim"]}d x {config["depth"]}L, {len(saved["part_b"])} programs)', fontsize = 10)
        ax.grid(alpha = 0.3)
        ax.legend(fontsize = 7, loc = 'best')

    fig.suptitle('learnability of discovered programs over time')
    fig.tight_layout()
    fig.savefig(path, dpi = 150)
    plt.close(fig)

if __name__ == '__main__':
    Fire(score)
