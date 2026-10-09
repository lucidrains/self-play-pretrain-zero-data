"""appendix C family detection over the appendix E tape

tape = first T emitted bytes, zero padded to T; up to 30 unrelated leading bytes,
then the remainder must satisfy the family recurrence mod 256 with minimal period
at least 30 over the remainder and its trailing window. a suffix of a valid
recurrence is valid, so checking the longest allowed stripped remainder covers
every start in 0..30
"""

from __future__ import annotations

from random import Random

from self_play_pretrain_zero_data.executors.base import exists

MOD = 256
MAX_LEADING = 30
MIN_PERIOD = 30
MIN_TERMS = 48
FAMILIES = ('arithmetic', 'quadratic', 'cubic', 'fibonacci', 'geometric')

def first(iterable):
    return next(iter(iterable), None)

def small_period(seq):
    # some p < MIN_PERIOD is a full period

    for p in range(1, min(MIN_PERIOD, len(seq))):
        if all(seq[i] == seq[i + p] for i in range(len(seq) - p)):
            return True

    return False

def constant_diffs(seq, order):
    # the order-th finite difference is constant mod 256

    d = list(seq)

    for _ in range(order):
        if len(d) < 2:
            return False

        d = [(d[i + 1] - d[i]) % MOD for i in range(len(d) - 1)]

    return all(x == d[0] for x in d)

def fibonacci_like(seq):
    # v_n = v_{n-1} + v_{n-2} mod 256, arbitrary seed pair

    return all((seq[i - 2] + seq[i - 1]) % MOD == seq[i] for i in range(2, len(seq)))

def geometric_like(seq):
    # v_n = r v_{n-1} mod 256 for some integer ratio

    if len(seq) < 2:
        return False

    ratios = [r for r in range(MOD) if (r * seq[0]) % MOD == seq[1]]

    return any(all((r * seq[i - 1]) % MOD == seq[i] for i in range(1, len(seq))) for r in ratios)

def assign(seq):
    # lowest order family satisfied, constant sequences land as arithmetic and the period guard drops them

    if constant_diffs(seq, 1):
        return 'arithmetic'
    if constant_diffs(seq, 2):
        return 'quadratic'
    if constant_diffs(seq, 3):
        return 'cubic'
    if fibonacci_like(seq):
        return 'fibonacci'
    if geometric_like(seq):
        return 'geometric'

    return None

def detect(output, tape_len):
    v = [ord(c) for c in output[:tape_len]]
    v += [0] * (tape_len - len(v))

    remainder = v[min(MAX_LEADING, tape_len - 1):]

    if len(remainder) < MIN_TERMS or small_period(remainder) or small_period(remainder[-2 * MIN_PERIOD:]):
        return set()

    return {family} if exists(family := assign(remainder)) else set()

def family_of(output, tape_len):
    hits = detect(output, tape_len)
    return first(family for family in FAMILIES if family in hits)

# uniform prior baseline, sample the augmented alphabet until the halt symbol

def uniform_baseline(executor, num_samples, seed, max_length = 64):
    rng = Random(seed)
    counts = dict.fromkeys(FAMILIES, 0)
    halt = getattr(executor, 'halt_symbol', None)

    for index in range(num_samples):
        program = ''

        for _ in range(max_length):
            char = rng.choice(executor.alphabet)
            program += char

            if exists(halt) and char == halt:
                break

        for family in detect(executor.execute(program, seed = seed).output, executor.max_output_len):
            counts[family] += 1

        if num_samples >= 100_000 and (index + 1) % 100_000 == 0:
            print(f'  uniform baseline {index + 1}/{num_samples} -> {counts}', flush = True)

    return counts

def expected_first_round(hits, num_samples, per_round):
    # first appearance under the uniform prior at the same programs / round, zero hits use the rule of three p <= 3 / num_samples

    if num_samples <= 0:
        return None, True

    rate = hits / num_samples if hits > 0 else 3. / num_samples

    return 1. / (rate * per_round), hits == 0
