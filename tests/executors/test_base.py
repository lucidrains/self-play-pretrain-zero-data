from self_play_pretrain_zero_data import Executor, SelfPlay

class BCT(Executor):
    def __init__(self, max_steps = 1000):
        self.max_steps = max_steps

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
    self_play = SelfPlay(generator = None, learner = None, executor = BCT())

    assert self_play.execute(['0', '01'], input = '10') == ['10', '1']
