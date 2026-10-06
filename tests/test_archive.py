import torch

from self_play_pretrain_zero_data import Brainfuck, Forth, NeuralCellularAutomata
from self_play_pretrain_zero_data.archive import (
    QualityDiversityArchive,
    crossover,
    delete,
    execution_loops_descriptor,
    execution_steps_descriptor,
    insert,
    program_length_descriptor,
    substitute,
)

def test_descriptors_are_derived_from_execution():
    info = Brainfuck().execute('+++[-].')

    assert program_length_descriptor(info) == 7
    assert execution_steps_descriptor(info) == 11
    assert execution_loops_descriptor(info) == 2

def test_custom_descriptor_fns():
    # test passing a single callable
    archive = QualityDiversityArchive(descriptor_fns = (lambda execution: len(execution.output)))

    archive.add('+++.', reward = 1., execution_info = Brainfuck().execute('+++.'))
    archive.add('++++.+.', reward = 1., execution_info = Brainfuck().execute('++++.+.'))  # different output length

    assert set(archive.archive) == {(1,), (2,)}

def test_program_outputs_are_archived():
    brainfuck_cases = (
        ('+.', '\x01'),
        ('+++.', '\x03'),
        ('+++++[->+<]>.', '\x05'),
    )

    forth_cases = (
        ('1 2 + EMIT', '\x03'),
        ('65 EMIT 66 EMIT', 'AB'),
        ('3 BEGIN DUP EMIT 1 - DUP 0= UNTIL', '\x03\x02\x01'),
    )

    nca_cases = (
        ('+1;11/11;21/12;22/22', '32/23'),
        ('-1;11/11;21/12;22/22', '21/12'),
    )

    for executor, cases in ((Brainfuck(), brainfuck_cases), (Forth(), forth_cases), (NeuralCellularAutomata(), nca_cases)):
        archive = QualityDiversityArchive(executor = executor)

        for program, expected_output in cases:
            assert executor(program) == expected_output
            assert archive.add(program, reward = len(expected_output))

        assert len(archive) == len(cases)
        assert {entry.program: entry.output for entry in archive} == dict(cases)

def test_add():
    archive = QualityDiversityArchive(descriptor_fns = (lambda execution: 0,))

    archive.add('+.', 1.5)                                                  # positional reward
    archive.add('++.', reward = 2.5)                                        # keyword reward
    archive.add('+++.', 3.5, execution_info = Brainfuck().execute('+++.'))  # precomputed execution info

    assert {entry.program: entry.reward for entry in archive} == {'+.': 1.5, '++.': 2.5, '+++.': 3.5}
    assert next(entry for entry in archive if entry.program == '+++.').output == '\x03'

def test_niche_keeps_only_the_best():
    archive = QualityDiversityArchive(
        descriptor_fns = (lambda execution: 0,),
        max_programs_per_niche = 2
    )

    archive.add('+.', reward = 1.)
    archive.add('+++.', reward = 3.)
    archive.add('+++++.', reward = 2.)

    assert len(archive) == 2
    assert {entry.program for entry in archive} == {'+++.', '+++++.'}

def test_resubmitting_a_program_updates_the_elite():
    archive = QualityDiversityArchive(descriptor_fns = (lambda execution: 0,))

    assert archive.add('+.', reward = 1.)
    assert archive.add('+.', reward = 2.)

    assert len(archive) == 1
    assert next(iter(archive)).reward == 2.

def test_remove_program():
    archive = QualityDiversityArchive()

    archive.add('+.', reward = 1.)
    archive.remove('+.')

    assert len(archive) == 0

def test_age_expiration_decays_and_evicts():
    archive = QualityDiversityArchive(reward_decay = 0.5, max_age = 2)

    archive.add('+.', reward = 4.)

    archive.advance_age()

    entry = next(iter(archive))

    assert entry.age == 1
    assert entry.reward == 2.

    archive.advance_age()

    assert len(archive) == 0

def test_mutation_produces_executable_programs():
    cases = (
        (Brainfuck(), '+++[->+<].'),
        (Forth(), '1 2 + EMIT'),
        (NeuralCellularAutomata(), '+1;11/11;21/12;22/22')
    )

    for executor, program in cases:
        archive = QualityDiversityArchive(executor = executor)

        mutants = [archive.mutate(program) for _ in range(32)]

        assert any(mutant != program for mutant in mutants)

        for mutant in mutants:
            executor.encode(mutant)  # raises if a token falls outside the alphabet

        # also test empty program mutation
        empty_mutant = archive.mutate('')
        assert len(empty_mutant) > 0
        executor.encode(empty_mutant)

def test_mutation_and_crossover():
    executor = Brainfuck()
    archive = QualityDiversityArchive(executor = executor, mutations = {insert: 1.})

    assert not archive.add('+.', reward = 0.)
    assert archive.add('+++.', reward = 1.)
    assert archive.add('+++++.', reward = 1.)

    # mutation - parent sampled from the archive, the weighted palette makes for a single insertion

    mutant = archive.mutate()
    assert len(mutant) in (5, 7)  # one of the two parents, one token inserted

    # crossover - parents sampled from the archive when not given, k points, one by default

    executor.execute(mutant)
    executor.execute(archive.crossover())
    executor.execute(archive.crossover(num_points = 3))

def test_encode_decode_live_on_the_archive():
    executor = Brainfuck()
    archive = QualityDiversityArchive(executor = executor)

    ids = archive.encode('+++[-].')

    assert ids.dtype == torch.long
    assert torch.equal(ids, torch.tensor(executor.encode('+++[-].')))
    assert archive.decode(ids) == '+++[-].'

def test_mutation_operators_operate_on_tensor_ids():
    executor = Brainfuck()
    archive = QualityDiversityArchive(executor = executor)

    ids = archive.encode('+++[-].')

    for mutation_fn in (substitute, insert, delete):
        mutant = mutation_fn(ids, executor.num_tokens)

        assert mutant.dtype == torch.long
        executor.encode(archive.decode(mutant))  # raises if a token falls outside the alphabet

def test_entries_cache_ids_and_accept_tensor_parents():
    executor = Brainfuck()
    archive = QualityDiversityArchive(executor = executor)

    assert archive.add('+++[-].', reward = 1.)

    ids = next(iter(archive)).ids

    assert torch.equal(ids, archive.encode('+++[-].'))

    executor.execute(archive.mutate(ids))
    executor.execute(archive.crossover(ids, ids, num_points = 2))

def test_standalone_encode_decode_fns():
    executor = Brainfuck()
    archive = QualityDiversityArchive(
        encode_fn = executor.encode,
        decode_fn = executor.decode,
        num_tokens = executor.num_tokens,
        mutations = {insert: 1.}
    )

    assert archive.add('++.', reward = 1.)
    assert len(archive.mutate()) == 4  # one insertion onto `++.`

def test_crossover_recombines_tensor_ids():
    ids_a = torch.tensor([1, 2, 3])
    ids_b = torch.tensor([4, 5, 6, 7])

    assert torch.equal(crossover(ids_a, ids_b, num_points = 0), ids_a)  # zero cuts return the first parent

    # one cut splices a prefix of a with a suffix of b

    prefixes_and_suffixes = {
        tuple(torch.cat((ids_a[:i], ids_b[j:])).tolist())
        for i in range(ids_a.numel() + 1)
        for j in range(ids_b.numel() + 1)
    }

    for _ in range(32):
        assert tuple(crossover(ids_a, ids_b).tolist()) in prefixes_and_suffixes
