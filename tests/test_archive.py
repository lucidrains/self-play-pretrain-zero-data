from self_play_pretrain_zero_data import Brainfuck, Forth
from self_play_pretrain_zero_data.archive import (
    QualityDiversityArchive,
    execution_loops_descriptor,
    execution_steps_descriptor,
    program_length_descriptor,
)

def test_descriptors_are_derived_from_execution():
    info = Brainfuck().execute('+++[-].')

    assert program_length_descriptor(info) == 7
    assert program_length_descriptor('+++[-].') == 7
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

    for executor, cases in ((Brainfuck(), brainfuck_cases), (Forth(), forth_cases)):
        archive = QualityDiversityArchive(executor = executor)

        for program, expected_output in cases:
            assert executor(program) == expected_output
            assert archive.add(program, reward = len(expected_output))

        assert len(archive) == len(cases)
        assert {entry.program: entry.output for entry in archive} == dict(cases)

def test_add_calling_conventions():
    # positional reward
    archive1 = QualityDiversityArchive(descriptor_fns = (lambda execution: 0,))
    archive1.add('+.', 1.5)
    assert len(archive1) == 1
    assert next(iter(archive1)).reward == 1.5

    # positional output and reward
    archive2 = QualityDiversityArchive(descriptor_fns = (lambda execution: 0,))
    archive2.add('+.', '\x01', 1.5)
    assert len(archive2) == 1
    entry2 = next(iter(archive2))
    assert entry2.program == '+.'
    assert entry2.output == '\x01'
    assert entry2.reward == 1.5

    # keyword reward
    archive3 = QualityDiversityArchive(descriptor_fns = (lambda execution: 0,))
    archive3.add('+.', reward = 1.5)
    assert next(iter(archive3)).reward == 1.5

    # precomputed execution info
    archive4 = QualityDiversityArchive(descriptor_fns = (lambda execution: 0,))
    archive4.add('+.', reward = 1.5, execution_info = Brainfuck().execute('+.'))
    entry4 = next(iter(archive4))
    assert entry4.output == '\x01'
    assert entry4.reward == 1.5

def test_derive_descriptors_from_string():
    archive = QualityDiversityArchive(executor = Brainfuck())
    assert archive.derive_descriptors('+++[-].') == (7, 2)

    # without executor, falls back to string program length
    archive_no_exec = QualityDiversityArchive(descriptor_fns = (program_length_descriptor,))
    assert archive_no_exec.derive_descriptors('+++[-].') == (7,)

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
    for executor, program in ((Brainfuck(), '+++[->+<].'), (Forth(), '1 2 + EMIT')):
        archive = QualityDiversityArchive(executor = executor)

        mutants = [archive.mutate(program) for _ in range(32)]

        assert any(mutant != program for mutant in mutants)

        for mutant in mutants:
            executor.encode(mutant)  # raises if a token falls outside the alphabet

        # also test empty program mutation
        empty_mutant = archive.mutate('')
        assert len(empty_mutant) > 0
        executor.encode(empty_mutant)
