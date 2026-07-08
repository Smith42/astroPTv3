"""Pythia checkpoint schedule (canonical implementation).

Pythia saves log2-spaced checkpoints early (steps 1, 2, 4, ..., 512) and then
every ``interval`` steps (1000 in the paper). The nanotron fork's trainer
lazy-imports :func:`should_checkpoint` when
``checkpoints.checkpoint_schedule: pythia`` is set, so the schedule stays in
this package (CPU-testable, single source of truth) and the fork stays thin.
"""

POW2_MAX = 512


def should_checkpoint(step: int, interval: int) -> bool:
    """True iff ``step`` is on the Pythia schedule: a power of two <= 512,
    or a positive multiple of ``interval``."""
    if step <= 0:
        return False
    if step <= POW2_MAX and step & (step - 1) == 0:
        return True
    return step % interval == 0


def checkpoint_steps(train_steps: int, interval: int) -> list[int]:
    """All scheduled steps in ``[1, train_steps]``, ascending."""
    return [s for s in range(1, train_steps + 1) if should_checkpoint(s, interval)]
