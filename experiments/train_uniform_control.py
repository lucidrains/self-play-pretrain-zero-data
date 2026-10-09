"""control learner trained from scratch on uniform random programs - the paper's uniform ablation

  python experiments/train_uniform_control.py --steps 968 --out experiments/runs/uniform-control
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from random import Random

import torch
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

def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--steps', type = int, default = 968)
    parser.add_argument('--batch-size', type = int, default = 64)
    parser.add_argument('--max-output', type = int, default = 112)
    parser.add_argument('--max-steps', type = int, default = 100_000)
    parser.add_argument('--dim', type = int, default = 192)
    parser.add_argument('--depth', type = int, default = 3)
    parser.add_argument('--heads', type = int, default = 6)
    parser.add_argument('--dim-head', type = int, default = 32)
    parser.add_argument('--lr', type = float, default = 3e-4)
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--threads', type = int, default = 8)
    parser.add_argument('--out', type = str, default = 'experiments/runs/uniform-control')

    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)

    out = Path(args.out)
    out.mkdir(parents = True, exist_ok = True)

    executor = Brainfuck(max_output_len = args.max_output, max_steps = args.max_steps)
    learner = Transformer(num_tokens = 256 + 1, dim = args.dim, depth = args.depth, dim_head = args.dim_head, heads = args.heads)
    optimizer = AdamW(learner.parameters(), lr = args.lr, weight_decay = 0.01)
    rng = Random(args.seed)

    print(f'learner {learner.num_parameters / 1e6:.2f}M params', flush = True)

    for step in range(args.steps):
        programs = uniform_programs(executor, args.batch_size, rng)
        outputs = [executor.execute(program).output for program in programs]
        ids = char_encode(outputs)

        loss = learner(ids, return_loss = True, reduce_loss = True)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(learner.parameters(), 1.)
        optimizer.step()

        if (step + 1) % 100 == 0:
            print(f'step {step + 1}/{args.steps} loss {loss.item():.4f}', flush = True)

    torch.save(learner.state_dict(), out / 'learner.pt')
    (out / 'results.json').write_text(json.dumps(dict(config = vars(args), steps = args.steps), indent = 2))
    print(f'wrote {out}/learner.pt')

if __name__ == '__main__':
    main()
