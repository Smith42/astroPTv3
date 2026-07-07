"""AstroPT3: SmolLM3-architecture multimodal astronomical foundation models."""

from transformers import AutoConfig, AutoModel

from .configuration_astropt3 import DEFAULT_MODALITIES, AstroPT3Config
from .modalities import Decoder, Encoder, ModalityConfig, ModalityRegistry, PositionEmbedder
from .modeling_astropt3 import AstroPT3Model, AstroPT3Output

AutoConfig.register("astropt3", AstroPT3Config)
AutoModel.register(AstroPT3Config, AstroPT3Model)

__all__ = [
    "AstroPT3Config",
    "AstroPT3Model",
    "AstroPT3Output",
    "DEFAULT_MODALITIES",
    "ModalityConfig",
    "ModalityRegistry",
    "Encoder",
    "Decoder",
    "PositionEmbedder",
]
