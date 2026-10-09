"""zero-shot bits per byte of a trained learner on a held-out byte corpus

  python experiments/transfer_eval.py --learner RUN/learner.pt --results RUN/results.json --text corpus.txt
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from fire import Fire

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

def evaluate(
    learner,
    text,
    results = None,
    threads = 8
):
    torch.set_num_threads(threads)

    results_path = Path(results) if results else Path(learner).parent / 'results.json'
    config = json.loads(results_path.read_text())['config']

    model = Transformer(
        num_tokens = 256 + 1,
        dim = config['dim'],
        depth = config['depth'],
        dim_head = config['dim_head'],
        heads = config['heads']
    )

    model.load_state_dict(torch.load(learner, map_location = 'cpu', weights_only = True))

    data = Path(text).read_bytes()
    ids = torch.tensor([byte + 1 for byte in data], dtype = torch.long)

    print(f'{Path(learner).parent.name:20s} {bits_per_byte(model, ids):.4f} bits/byte')

if __name__ == '__main__':
    Fire(evaluate)
