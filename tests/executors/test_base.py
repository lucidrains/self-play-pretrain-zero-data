import torch

from self_play_pretrain_zero_data import Executor, SelfPlay
from self_play_pretrain_zero_data.self_play import Transformer

class BCT(Executor):
    def __init__(self, max_steps = 1000):
        self.max_steps = max_steps
        self.num_tokens = 3

    def encode(self, program):
        return [int(c) + 1 for c in program if c in '01']

    def decode(self, ids):
        return ''.join('01'[i - 1] for i in ids if i != self.sos_eos_id and i != self.pad_id)

    def __call__(self, program, input = '', seed = None):
        code = [c for c in program if c in '01']
        data = [c for c in input if c in '01']

        out, pc = '', 0

        for _ in range(self.max_steps):
            if not code or not data:
                break

            if code[pc] == '0':
                out += data.pop(0)
                pc = (pc + 1) % len(code)
            else:
                if data[0] == '1':
                    data.append(code[(pc + 1) % len(code)])
                pc = (pc + 2) % len(code)

        return out

def test_bct():
    assert BCT(max_steps = 11)('00111', input = '101') == '10110'
    assert BCT()('') == ''

def test_self_play_accepts_contrived_executor():
    executor = BCT()

    generator = Transformer(num_tokens = executor.num_tokens, dim = 16, depth = 1, dim_head = 8, heads = 2, sos_eos_id = executor.sos_eos_id)
    learner = Transformer(num_tokens = 256, dim = 16, depth = 1, dim_head = 8, heads = 2)

    self_play = SelfPlay(generator = generator, learner = learner, executor = executor)

    loss, tangent = self_play(batch_size = 2, max_length = 4, verbose = False)

    assert tangent.shape == (2,)
