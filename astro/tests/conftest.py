from pathlib import Path

import pytest
import torch

from astropt3.config_io import load_model_config
from astropt3.data.packing import ObjectSequencer, PackedCollator
from astropt3.data.synthetic import make_record

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "model"


@pytest.fixture(scope="session")
def tiny_config():
    config, _ = load_model_config(CONFIG_DIR / "test-tiny.yaml")
    return config


@pytest.fixture(scope="session")
def sequencer(tiny_config):
    return ObjectSequencer(tiny_config)


@pytest.fixture(scope="session")
def collator(tiny_config):
    return PackedCollator(tiny_config, seq_len=896)


@pytest.fixture(scope="session")
def full_record():
    # seed 3 has both image and spectrum
    record = make_record(3)
    assert "spectrum" in record
    return record


@pytest.fixture(scope="session")
def image_only_record():
    # scan for a deterministic image-only record
    for i in range(64):
        record = make_record(i)
        if "spectrum" not in record:
            return record
    raise RuntimeError("no image-only record found in first 64 seeds")


@pytest.fixture(scope="session")
def tiny_model(tiny_config):
    torch.manual_seed(0)
    from astropt3 import AstroPT3Model

    model = AstroPT3Model(tiny_config)
    model.eval()
    return model
