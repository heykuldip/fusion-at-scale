"""The sampler checks in one process: each simulated process's sampler is built with an explicit rank, so no process
group is needed.
"""
import pytest
from torch.utils.data.distributed import DistributedSampler

from fusion_scale.data.sampler_checks import SEED, IndexDataset, epoch_indices, expected, summarize

CASES = [(n, w, dl) for n in (12, 13, 10, 7) for w in (2, 3, 4) for dl in (False, True)]


def samplers(n, w, drop_last):
    """One sampler per simulated process, as each process would build its own."""
    return [DistributedSampler(IndexDataset(n), num_replicas=w, rank=r, shuffle=True, seed=SEED, drop_last=drop_last)
            for r in range(w)]


@pytest.mark.parametrize('n,w,drop_last', CASES)
def test_one_epoch_matches_docs(n, w, drop_last):
    per_rank = [epoch_indices(s, 0, call_set_epoch=True) for s in samplers(n, w, drop_last)]
    assert summarize(per_rank, n) == expected(n, w, drop_last)


def test_set_epoch_changes_the_order_and_its_absence_repeats_it():
    """With set_epoch, epochs differ; without it, the order repeats."""
    n, w, drop_last = 13, 3, False

    changing = samplers(n, w, drop_last)
    epoch0 = [epoch_indices(s, 0, True) for s in changing]
    epoch1 = [epoch_indices(s, 1, True) for s in changing]
    assert epoch0 != epoch1

    fixed = samplers(n, w, drop_last)
    epoch0 = [epoch_indices(s, 0, False) for s in fixed]
    epoch1 = [epoch_indices(s, 1, False) for s in fixed]
    assert epoch0 == epoch1


def test_drop_last_without_set_epoch_never_sees_the_same_tail():
    """Without set_epoch, the same samples are omitted in every epoch; with it, the omitted samples change."""
    n, w, drop_last = 7, 4, True                      # each epoch drops 3 of 7

    fixed = samplers(n, w, drop_last)
    epoch0 = [epoch_indices(s, 0, False) for s in fixed]
    epoch1 = [epoch_indices(s, 1, False) for s in fixed]
    assert epoch0 == epoch1

    missing0 = set(range(n)) - {i for rank in epoch0 for i in rank}
    missing1 = set(range(n)) - {i for rank in epoch1 for i in rank}
    assert missing0 == missing1
    assert missing0

    changing = samplers(n, w, drop_last)
    dropped = []
    for epoch in range(5):
        seen = {i for s in changing for i in epoch_indices(s, epoch, True)}
        dropped.append(frozenset(set(range(n)) - seen))
    # Probabilistic: each epoch drops a random 3 of 7 (35 possible sets), so all five matching by accident has a
    # chance of (1/35)^4, about 1 in 1.5 million. For a given seed and PyTorch version, the outcome is fixed.
    assert len(set(dropped)) > 1


def test_negative_control():
    """A deliberately incorrect expectation should fail."""
    n, w, drop_last = 13, 3, False
    per_rank = [
        epoch_indices(s, 0, True)
        for s in samplers(n, w, drop_last)
    ]
    actual = summarize(per_rank, n)

    # Incorrect claim: padding never repeats indices.
    wrong_expected = expected(n, w, drop_last).copy()
    wrong_expected["repeated"] = 0

    assert actual != wrong_expected
    assert actual["repeated"] > 0
