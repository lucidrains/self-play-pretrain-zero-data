import torch

from self_play_pretrain_zero_data import (
    Brainfuck,
    CheckpointReference,
    ProgramBatch,
    QualityDiversityArchive,
    SelfPlay,
    Transformer,
    fresh_programs,
    mutation_programs,
    replay_programs
)

def make_self_play(tmp_path, archive = None, proposals = None):
    executor = Brainfuck()

    generator = Transformer(num_tokens = executor.num_tokens, dim = 16, depth = 1, dim_head = 8, heads = 2)
    learner = Transformer(num_tokens = 256 + 1, dim = 16, depth = 1, dim_head = 8, heads = 2)

    return SelfPlay(
        generator = generator,
        learner = learner,
        executor = executor,
        archive = archive,
        proposals = proposals,
        learner_reference = CheckpointReference(folder = tmp_path)
    )

def test_program_pool_union(tmp_path):
    torch.manual_seed(0)

    executor = Brainfuck()
    archive = QualityDiversityArchive(executor = executor)
    archive.add('+++.', reward = 1., log_prob = -3.)

    self_play = make_self_play(tmp_path, archive = archive)

    fresh = fresh_programs(self_play, 2, max_length = 4, decode_fn = executor.decode)
    mutants = mutation_programs(self_play, 2)
    replays = replay_programs(self_play, 2)

    # fresh: policy gradient + archive, mutations: archive only, replays: policy gradient only

    assert fresh.pg_mask.all() and fresh.admit.all()
    assert not mutants.pg_mask.any() and mutants.admit.all()
    assert replays.pg_mask.all() and not replays.admit.any()

    assert torch.allclose(replays.old_log_probs, torch.full((2,), -3., device = replays.old_log_probs.device))

    batch = ProgramBatch.concat([fresh, mutants, replays])

    assert len(batch.programs) == 6
    assert batch.decoded_ids.shape == batch.seq_mask.shape
    assert (batch.seq_mask.sum(dim = -1) >= 1).all()

def test_custom_pool_proportions(tmp_path):
    counts = {}

    def custom_programs(self_play, num_programs, **generation_kwargs):
        counts['custom'] = num_programs
        return self_play.program_batch(['++.'] * num_programs)

    self_play = make_self_play(tmp_path, proposals = {'fresh': 1., custom_programs: 3.})

    batch = self_play.sample_pool(8)

    # names resolve through the pool dict, callables pass through, weights are exact shares

    assert counts == {'custom': 6}
    assert len(batch.programs) == 8
    assert batch.programs.count('++.') == 6

def test_mutation_rows_drop_out_of_policy_gradient(tmp_path):
    torch.manual_seed(0)

    self_play = make_self_play(tmp_path)

    replay_ids = torch.tensor([[0, 1, 2, 3], [0, 2, 1, 4], [0, 3, 3, 1]])
    seq_mask = torch.tensor([[False, True, True, True]]).repeat(3, 1)
    rewards = torch.tensor([1., 2., 3.])
    old_log_probs = torch.tensor([-1., -2., -3.])
    pg_mask = torch.tensor([True, False, True])

    loss, _, _ = self_play.grpo_loss(rewards, old_log_probs = old_log_probs, replay_ids = replay_ids, prompt_len = 1, seq_mask = seq_mask, pg_mask = pg_mask)

    # perturbing the mutation row leaves the policy gradient untouched

    perturbed = old_log_probs.clone()
    perturbed[~pg_mask] = 1e3

    perturbed_loss, _, _ = self_play.grpo_loss(rewards, old_log_probs = perturbed, replay_ids = replay_ids, prompt_len = 1, seq_mask = seq_mask, pg_mask = pg_mask)

    assert torch.allclose(loss, perturbed_loss)

def test_self_play_fills_pool_and_updates_archive(tmp_path):
    torch.manual_seed(0)

    executor = Brainfuck()
    archive = QualityDiversityArchive(executor = executor)

    self_play = make_self_play(tmp_path, archive = archive)

    num_epochs = 2

    losses, _ = self_play(batch_size = 4, max_length = 4, verbose = False, epochs = num_epochs, decode_fn = executor.decode)

    assert losses.shape == (num_epochs,)

    # empty archive, fresh fills the pool, then admitted elites seed mutations and replays

    assert len(archive) > 0
    assert all(entry.log_prob is not None for entry in archive)
    assert all(entry.reward > 0. for entry in archive)

def test_program_batch_seq_mask_and_edge_cases(tmp_path):
    self_play = make_self_play(tmp_path)

    batch = self_play.program_batch(['++.'])

    # prompt token (sos) at index 0 is excluded from seq_mask, matching generate

    assert batch.decoded_ids[:, 0].item() == self_play.executor.sos_eos_id
    assert not batch.seq_mask[:, 0].item()
    assert batch.seq_mask.sum().item() == 3

    # empty program string does not crash on embedding or cat

    empty_str_batch = self_play.program_batch([''])
    assert empty_str_batch.decoded_ids.dtype == torch.long
    assert empty_str_batch.seq_mask.sum().item() == 0

    # empty list returns None

    assert self_play.program_batch([]) is None

def test_proposals_sequence_normalized(tmp_path):
    self_play = make_self_play(tmp_path, proposals = ('fresh', 'mutation', 'replay'))

    assert self_play.proposals == dict(fresh = 1., mutation = 1., replay = 1.)

def test_crossover_pool(tmp_path):
    torch.manual_seed(0)

    executor = Brainfuck()
    archive = QualityDiversityArchive(executor = executor)
    archive.add('+++.', reward = 1., log_prob = -3.)
    archive.add('---.', reward = 2., log_prob = -4.)

    self_play = make_self_play(tmp_path, archive = archive, proposals = {'fresh': 1., 'crossover': 1.})

    batch = self_play.sample_pool(4)

    assert len(batch.programs) == 4
    assert batch.pg_mask.sum() == 2
    assert batch.admit.all()
