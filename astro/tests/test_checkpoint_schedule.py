"""Pythia checkpoint schedule: exact membership and step enumeration."""

from astropt3.checkpoint_schedule import checkpoint_steps, should_checkpoint

PYTHIA_1000 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, 2000]


def test_exact_membership_interval_1000():
    expected = set(PYTHIA_1000)
    for step in range(-5, 2049):
        assert should_checkpoint(step, 1000) == (step in expected), step


def test_checkpoint_steps_enumeration():
    assert checkpoint_steps(2000, 1000) == PYTHIA_1000
    assert checkpoint_steps(137, 1000) == [1, 2, 4, 8, 16, 32, 64, 128]


def test_interval_composes_below_512():
    # a multiple of the interval is saved even inside the log2 region
    assert should_checkpoint(100, 100)
    assert not should_checkpoint(100, 1000)
