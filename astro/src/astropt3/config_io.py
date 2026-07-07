"""Load model-size and data YAMLs."""

import os
from pathlib import Path

import yaml

from .configuration_astropt3 import AstroPT3Config

# astro/src/astropt3/config_io.py -> repo root (parent of astro/).
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Keys that describe the run rather than the architecture.
_META_KEYS = {"name", "nominal_params"}


def load_model_config(path: str | Path) -> tuple[AstroPT3Config, dict]:
    """Read a configs/model/*.yaml file -> (AstroPT3Config, meta dict)."""
    raw = yaml.safe_load(Path(path).read_text())
    meta = {k: raw[k] for k in _META_KEYS if k in raw}
    arch = {k: v for k, v in raw.items() if k not in _META_KEYS}
    return AstroPT3Config(**arch), meta


def load_data_config(path: str | Path) -> dict:
    """Read a configs/data/*.yaml file -> plain dict."""
    return yaml.safe_load(Path(path).read_text())


def resolve_data_root(data_config: dict) -> Path:
    """Absolute path to this config's prepared-data root.

    Precedence: the ``ASTROPT3_DATA_ROOT`` env var, else the config's
    ``paths.root``. ``$VARS`` and ``~`` are expanded; a relative result is
    resolved against the repo root (the parent of ``astro/``), so the default
    ``../astroPTv3_data/pilot_v1`` sits beside the repo instead of being tied
    to any one machine. Set ``ASTROPT3_DATA_ROOT`` to place the corpus
    elsewhere (e.g. fast scratch on the training cluster).
    """
    root = os.environ.get("ASTROPT3_DATA_ROOT") or data_config["paths"]["root"]
    root = Path(os.path.expandvars(root)).expanduser()
    if not root.is_absolute():
        root = _REPO_ROOT / root
    return root


def sequencer_kwargs_from_data_config(data_config: dict) -> dict:
    """Asinh-stretch kwargs for ``ObjectSequencer`` from a data config dict.

    Empty until ``scripts/compute_norm_stats.py`` has filled the
    ``normalization`` block (the sequencer then falls back to plain asinh).
    """
    norm = data_config.get("normalization") or {}
    if norm.get("image_p99") is None:
        return {}
    return {
        "image_p1": norm["image_p1"],
        "image_p99": norm["image_p99"],
        "alpha": norm.get("asinh_alpha", 20.0),
    }
