"""AstroPT3 configuration: a SmolLM3 body with continuous-modality I/O."""

from transformers.models.smollm3.configuration_smollm3 import SmolLM3Config

from .data.band_registry import _DIV_FACTOR
from .data.spectral import _DIV_FACTOR as _SPECTRA_DIV_FACTOR
from .modalities import ModalityRegistry
from .tokenization import VOCAB_SIZE

# Pilot modalities, pinned to the verified MMU schemas (see plan):
# - images:  (3, 152, 152) flux cubes, center crop 96x96, patch 8 -> 144
#   tokens of 8*8*3 = 192 (max_positions 361 is a harmless ceiling)
# - spectra: 7781-bin DESI spectra, patch 256 -> 31 tokens; position = per-patch
#   mean wavelength, normalized, projected by an affine PositionEmbedder
# - Z / ebv / photometry: ADR 0008 scalar modalities — one-token spans over
#   the catalog scalars the records already carry, GMM-headed under both
#   tokenisers. loss_weight 0.1 keeps the objective dominated by images and
#   spectra (the per-token mean that balanced 364-vs-31 tokens creates the
#   imbalance at 144:1 — the ADR's scoped override of 0005's 1:1 principle).
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
    {
        "name": "Z",
        "input_size": 1,
        "patch_size": 1,
        "pos_type": "index",
        "max_positions": 1,
        "loss_weight": 0.1,
        "scalar": True,
    },
    {
        "name": "ebv",
        "input_size": 1,
        "patch_size": 1,
        "pos_type": "index",
        "max_positions": 1,
        "loss_weight": 0.1,
        "scalar": True,
    },
    {
        # one joint 3-dim span (g, r, z), not three modalities: colour is
        # the physical quantity, and a joint GMM models it directly
        "name": "photometry",
        "input_size": 3,
        "patch_size": 1,
        "pos_type": "index",
        "max_positions": 1,
        "loss_weight": 0.1,
        "scalar": True,
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
        jetformer_noise_max: float = 0.1,
        jetformer_noise_min: float = 0.0,
        scalar_gmm_k: int = 5,
        huber_delta: float = 1.0,
        special_token_ce_weight: float = 0.0,
        image_norm_divisor: float = _DIV_FACTOR,
        spectra_norm_divisor: float = _SPECTRA_DIV_FACTOR,
        spiral: bool = True,
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
        self.jetformer_noise_max = jetformer_noise_max
        self.jetformer_noise_min = jetformer_noise_min
        # mixture count of the ADR 0008 scalar GMM heads (photometric-redshift
        # posteriors are multimodal; K=5 is the ADR's unswept starting point)
        self.scalar_gmm_k = scalar_gmm_k
        self.huber_delta = huber_delta
        self.special_token_ce_weight = special_token_ce_weight
        # arcsinh divisor of the physical image normalization; consumed by
        # ObjectSequencer (forward) and scripts/generate.py (inverse), so a
        # checkpoint always normalizes and inverts with the divisor it
        # trained with (the default back-fills configs/checkpoints saved
        # before the field existed — note pre-physical-norm PU-asinh
        # checkpoints are incompatible regardless, see docs)
        self.image_norm_divisor = image_norm_divisor
        # arcsinh knee (nMgy) of the physical spectra normalization (ADR
        # 0007), the spectra counterpart of image_norm_divisor: consumed by
        # ObjectSequencer (forward) and eval/samples.py (inverse). Spectra
        # checkpoints saved before the field existed trained on raw DESI
        # flux and are incompatible regardless of the back-fill — retrain.
        self.spectra_norm_divisor = spectra_norm_divisor
        # center-outward spiral patch order for image tokens (ADR 0004).
        # The field is the single source of truth for the order a checkpoint
        # trained in: ObjectSequencer spiralises iff it is True, and the
        # inverse path (eval/samples.py) antispiralises iff the LOADED
        # checkpoint's config says True. The __init__ default is True so every
        # run (and any config/older fork config missing the field) is spiral
        # by default; all configs/model + configs/nanotron YAMLs set the field
        # explicitly. NOTE: this flips the old back-fill from raster to spiral,
        # so any raster checkpoint saved before the field existed (the 70M/160M
        # raster shakeouts) now loads as spiral and decodes scrambled — retrain
        # or load those with `spiral: false` passed explicitly.
        self.spiral = spiral
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
