"""Load model-size YAMLs into AstroPT3Config objects."""

from pathlib import Path

import yaml

from .configuration_astropt3 import AstroPT3Config

# Keys that describe the run rather than the architecture.
_META_KEYS = {"name", "nominal_params"}


def load_model_config(path: str | Path) -> tuple[AstroPT3Config, dict]:
    """Read a configs/model/*.yaml file -> (AstroPT3Config, meta dict)."""
    raw = yaml.safe_load(Path(path).read_text())
    meta = {k: raw[k] for k in _META_KEYS if k in raw}
    arch = {k: v for k, v in raw.items() if k not in _META_KEYS}
    return AstroPT3Config(**arch), meta
