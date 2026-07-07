from pathlib import Path

from astropt3.train_smoke import run

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "model" / "test-tiny.yaml"


def test_smoke_training_learns():
    losses = run(str(CONFIG), steps=40, objects_per_batch=2, seq_len=896, lr=1e-3)
    assert losses[-1] < 0.7 * losses[0], f"{losses[0]:.4f} -> {losses[-1]:.4f}"
