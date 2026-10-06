import torch

from self_play_pretrain_zero_data import Brainfuck, Forth, NeuralCellularAutomata
from self_play_pretrain_zero_data.archive import (
    QualityDiversityArchive,
    crossover,
    crossover_batch,
    delete,
    delete_batch,
    execution_loops_descriptor,
    execution_steps_descriptor,
    insert,
    insert_batch,
    program_length_descriptor,
    substitute,
    substitute_batch,
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

def test_batch_operators_agree_with_single_program_wrappers():
    pairs = (
        (substitute, substitute_batch),
        (insert, insert_batch),
        (delete, delete_batch),
    )

    for single_fn, batch_fn in pairs:
        ids = torch.tensor([1, 2, 3])
        mask = torch.ones((1, ids.numel()), dtype = torch.bool)

        single = single_fn(ids, 8, generator = torch.Generator().manual_seed(0))
        batched, batched_mask = batch_fn(ids[None], mask, 8, generator = torch.Generator().manual_seed(0))

        assert torch.equal(single, batched[0][batched_mask[0]])

    ids_a, ids_b = torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6, 7])
    mask_a = torch.ones((1, ids_a.numel()), dtype = torch.bool)
    mask_b = torch.ones((1, ids_b.numel()), dtype = torch.bool)

    single = crossover(ids_a, ids_b, num_points = 2, generator = torch.Generator().manual_seed(0))
    batched, batched_mask = crossover_batch(ids_a[None], mask_a, ids_b[None], mask_b, num_points = 2, generator = torch.Generator().manual_seed(0))

    assert torch.equal(single, batched[0])
    assert batched_mask.all()

def test_batched_mutation_operators_only_touch_valid_tokens():
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])

    generator = torch.Generator().manual_seed(0)
    sub_ids, sub_mask = substitute_batch(ids, mask, num_tokens = 16, generator = generator)

    assert torch.equal(sub_mask, mask)
    assert torch.equal(sub_ids[~mask], ids[~mask])  # padding is untouched
    assert ((sub_ids != ids) & mask).sum(dim = -1).tolist() == [1, 1]

    generator = torch.Generator().manual_seed(0)
    ins_ids, ins_mask = insert_batch(ids, mask, num_tokens = 16, generator = generator)

    assert ins_ids.shape == (2, 5)
    assert ins_mask.sum(dim = -1).tolist() == [4, 3]

    for row, length in enumerate(ins_mask.sum(dim = -1).tolist()):
        original = ids[row][mask[row]].tolist()
        mutant = ins_ids[row][:length].tolist()
        assert any(mutant[:i] + mutant[i + 1:] == original for i in range(len(mutant)))

    generator = torch.Generator().manual_seed(0)
    del_ids, del_mask = delete_batch(ids, mask, num_tokens = 16, generator = generator)

    assert del_mask.sum(dim = -1).tolist() == [2, 1]

    for row, length in enumerate(del_mask.sum(dim = -1).tolist()):
        original = ids[row][mask[row]].tolist()
        mutant = del_ids[row][:length].tolist()
        assert any(mutant == original[:i] + original[i + 1:] for i in range(len(original)))

def test_batched_operators_handle_empty_programs():
    ids = torch.tensor([[0, 0, 0], [5, 0, 0]])
    mask = torch.tensor([[False, False, False], [True, False, False]])

    generator = torch.Generator().manual_seed(0)
    _, sub_mask = substitute_batch(ids, mask, num_tokens = 8, generator = generator)

    generator = torch.Generator().manual_seed(0)
    _, del_mask = delete_batch(ids, mask, num_tokens = 8, generator = generator)

    assert sub_mask.sum(dim = -1).tolist() == [1, 1]  # an empty program degrades to an insertion
    assert del_mask.sum(dim = -1).tolist() == [1, 0]

    empty = ids[:, :0]
    out_ids, out_mask = crossover_batch(empty, empty, ids, mask, generator = generator)

    assert out_ids.shape[0] == 2
    assert (out_mask.sum(dim = -1) <= mask.sum(dim = -1)).all()

def test_batched_crossover_recombines_every_pair():
    ids_a = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    ids_b = torch.tensor([[6, 7, 8, 9], [10, 11, 12, 0]])
    mask_a, mask_b = ids_a != 0, ids_b != 0

    generator = torch.Generator().manual_seed(0)
    out_ids, out_mask = crossover_batch(ids_a, mask_a, ids_b, mask_b, num_points = 1, generator = generator)

    for row, length in enumerate(out_mask.sum(dim = -1).tolist()):
        parent_a = ids_a[row][mask_a[row]].tolist()
        parent_b = ids_b[row][mask_b[row]].tolist()
        offspring = out_ids[row][:length].tolist()

        assert any(offspring == parent_a[:i] + parent_b[j:] for i in range(len(parent_a) + 1) for j in range(len(parent_b) + 1))

def test_archive_batch_mutation_and_crossover():
    executor = Brainfuck()
    archive = QualityDiversityArchive(executor = executor)

    archive.add('+++.', reward = 1.)
    archive.add('+++++[->+<]>.', reward = 1.)

    mutants = archive.mutate_batch(16)
    assert len(mutants) == 16
    assert any(mutant != '+++.' and mutant != '+++++[->+<]>.' for mutant in mutants)

    for mutant in mutants:
        executor.encode(mutant)  # raises if a token falls outside the alphabet

    offspring = archive.crossover_batch(16, num_points = 2)
    assert len(offspring) == 16

    for program in offspring:
        executor.encode(program)
