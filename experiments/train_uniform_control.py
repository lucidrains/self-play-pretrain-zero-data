"""control learner trained from scratch on uniform random programs - the paper's uniform ablation

  python experiments/train_uniform_control.py --steps 968 --out experiments/runs/uniform-control
"""

from __future__ import annotations

import json
from pathlib import Path
from random import Random
from types import SimpleNamespace

import torch
from fire import Fire
from torch.optim import AdamW

from self_play_pretrain_zero_data import Brainfuck, Transformer
from self_play_pretrain_zero_data.self_play import char_encode


def uniform_programs(executor, count, rng, max_length = 32):
    programs = []

    for _ in range(count):
        program = ''

        for _ in range(max_length):
            char = rng.choice(executor.alphabet)
            program += char

            if char == executor.halt_symbol:
                break

        programs.append(program)

    return programs

def train(
    steps = 968,
    batch_size = 64,
    max_output = 112,
    max_steps = 100_000,
    dim = 192,
    depth = 3,
    heads = 6,
    dim_head = 32,
    lr = 3e-4,
    seed = 0,
    threads = 8,
    out = 'experiments/runs/uniform-control'
):
    config = SimpleNamespace(**locals())

    torch.set_num_threads(threads)
    torch.manual_seed(seed)

    out = Path(out)
    out.mkdir(parents = True, exist_ok = True)

    executor = Brainfuck(max_output_len = max_output, max_steps = max_steps)
    learner = Transformer(num_tokens = 256 + 1, dim = dim, depth = depth, dim_head = dim_head, heads = heads)
    optimizer = AdamW(learner.parameters(), lr = lr, weight_decay = 0.01)
    rng = Random(seed)

    print(f'learner {learner.num_parameters / 1e6:.2f}M params', flush = True)

    for step in range(steps):
        programs = uniform_programs(executor, batch_size, rng)
        outputs = [executor.execute(program).output for program in programs]
        ids = char_encode(outputs)

        loss = learner(ids, return_loss = True, reduce_loss = True)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(learner.parameters(), 1.)
        optimizer.step()

        if (step + 1) % 100 == 0:
            print(f'step {step + 1}/{steps} loss {loss.item():.4f}', flush = True)

    torch.save(learner.state_dict(), out / 'learner.pt')
    (out / 'results.json').write_text(json.dumps(dict(config = vars(config), steps = steps), indent = 2))
    print(f'wrote {out}/learner.pt')

if __name__ == '__main__':
    Fire(train)
