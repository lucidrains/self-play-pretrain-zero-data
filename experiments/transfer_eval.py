"""zero-shot bits per byte of a trained learner on a held-out byte corpus

  python experiments/transfer_eval.py --learner RUN/learner.pt --results RUN/results.json --text corpus.txt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from self_play_pretrain_zero_data import Transformer


@torch.no_grad()
def bits_per_byte(learner, ids, window = 512):
    learner.eval()

    totals, counts = 0., 0

    for start in range(0, len(ids) - window - 1, window):
        chunk = ids[start:start + window + 1][None]
        loss, mask = learner(chunk, return_loss = True)
        totals += loss.sum().item()
        counts += mask.sum().item()

    return totals / counts / math.log(2)

def main():
    parser = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--learner', type = str, required = True)
    parser.add_argument('--results', type = str)
    parser.add_argument('--text', type = str, required = True)
    parser.add_argument('--threads', type = int, default = 8)

    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    results_path = Path(args.results) if args.results else Path(args.learner).parent / 'results.json'
    config = json.loads(results_path.read_text())['config']

    learner = Transformer(
        num_tokens = 256 + 1,
        dim = config['dim'],
        depth = config['depth'],
        dim_head = config['dim_head'],
        heads = config['heads']
    )

    learner.load_state_dict(torch.load(args.learner, map_location = 'cpu', weights_only = True))

    data = Path(args.text).read_bytes()
    ids = torch.tensor([byte + 1 for byte in data], dtype = torch.long)

    print(f'{Path(args.learner).parent.name:20s} {bits_per_byte(learner, ids):.4f} bits/byte')

if __name__ == '__main__':
    main()
