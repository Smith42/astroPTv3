"""AstroPT3 configuration: a SmolLM3 body with continuous-modality I/O."""

from transformers.models.smollm3.configuration_smollm3 import SmolLM3Config

from .modalities import ModalityRegistry
from .tokenization import VOCAB_SIZE

# Pilot modalities, pinned to the verified MMU schemas (see plan):
# - images:  (3, 152, 152) flux cubes, patch 8 -> 361 tokens of 8*8*3 = 192
# - spectra: 7781-bin DESI spectra, patch 256 -> 31 tokens; position = per-patch
#   mean wavelength, normalized, projected by an affine PositionEmbedder
DEFAULT_MODALITIES = [
    {
        "name": "images",
        "input_size": 192,
        "patch_size": 8,
        "pos_type": "index",
        "max_positions": 361,
        "loss_weight": 1.0,
    },
    {
        "name": "spectra",
        "input_size": 256,
        "patch_size": 256,
        "pos_type": "continuous",
        "pos_input_size": 1,
        "loss_weight": 1.0,
    },
]


class AstroPT3Config(SmolLM3Config):
    model_type = "astropt3"

    def __init__(
        self,
        modalities: list[dict] | None = None,
        tokeniser: str = "affine",
        jetformer_flow_steps: int = 4,
        jetformer_flow_hidden: int = 128,
        jetformer_gmm_k: int = 4,
        huber_delta: float = 1.0,
        special_token_ce_weight: float = 0.0,
        vocab_size: int = VOCAB_SIZE,
        hidden_size: int = 512,
        intermediate_size: int = 1536,
        num_hidden_layers: int = 22,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 2,
        max_position_embeddings: int = 4096,
        rope_theta: float = 100_000.0,
        no_rope_layer_interval: int = 4,
        tie_word_embeddings: bool = False,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 1,
        **kwargs,
    ):
        self.modalities = modalities if modalities is not None else DEFAULT_MODALITIES
        self.tokeniser = tokeniser
        self.jetformer_flow_steps = jetformer_flow_steps
        self.jetformer_flow_hidden = jetformer_flow_hidden
        self.jetformer_gmm_k = jetformer_gmm_k
        self.huber_delta = huber_delta
        self.special_token_ce_weight = special_token_ce_weight
        kwargs["use_cache"] = False  # reload passes it back through kwargs
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            no_rope_layer_interval=no_rope_layer_interval,
            tie_word_embeddings=tie_word_embeddings,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )

    def modality_registry(self) -> ModalityRegistry:
        return ModalityRegistry(self.modalities)
