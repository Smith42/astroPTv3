# SmolLM AstroPT: Project Plan

## Overview

SmolLM AstroPT is a multimodal astronomical foundation model that follows the SmolVLM architecture as closely as possible, replacing natural-image components with astronomical data from the Multimodal Universe (MMU). The project trains models from ~135M to 3B parameters on 100TB+ of astronomical data, testing whether a simple, open, community-reproducible model can rival the performance of larger, bespoke efforts like AION-1.

The core thesis is the same one AstroPT has argued from the start: general architectures plus good data beat clever engineering. SmolLM AstroPT extends this to the multimodal setting.

---

## Architecture

SmolLM AstroPT follows SmolVLM's design: a pretrained vision encoder produces continuous visual tokens that are projected into a language model backbone via a lightweight connector. We extend this pattern to handle the full range of astronomical data modalities present in MMU.

### Backbone

SmolLM serves as the language model backbone. Target model sizes: 135M, 360M, 1.7B, and 3B parameters. The 3B variant matches AION-1's largest model in parameter count, enabling direct comparison.

### Modality-Specific Tokenization

Following SmolVLM, we avoid discrete tokenization (no VQ-VAE, no codebooks) and instead use continuous representations throughout. Each modality gets a minimal projection pipeline:

#### RGB and Multi-Band Images

- **RGB images**: SigLIP (pretrained, frozen or lightly tuned) → pixel shuffle (2×2) → MLP projection → SmolLM.
- **Multi-band FITS**: Learned linear channel projection (N-bands → 3 channels) → SigLIP → pixel shuffle (2×2) → MLP projection → SmolLM.

This follows SmolVLM exactly for RGB, with a single additional linear layer for FITS. No new vision encoder is trained. SigLIP is used off-the-shelf, consistent with the Platonic Representation Hypothesis (Duraphe, Smith et al., NeurIPS 2025): pretrained foundation models transfer meaningful representations to astronomical data.

SmolVLM found that replacing a 400M-parameter SigLIP with a 93M-parameter variant yielded only marginally worse results. We adopt the smaller SigLIP-base-patch16 for efficiency.

#### Spectra

- 1D patching (chunk spectrum into fixed-length windows) → **wavelength shuffle** → MLP projection → SmolLM.

**Wavelength shuffle** is the 1D analogue of pixel shuffle for images. Adjacent wavelength bins within each patch are stacked into the feature dimension, reducing token count while preserving all spectral information. For example, a 4× wavelength shuffle on a spectrum with 4000 bins produces 1000 tokens, each encoding 4 adjacent bins. This keeps spectral sequences from dominating the context window when concatenated with image and other modality tokens.

The shuffle ratio is a hyperparameter to ablate. Start with 4× and evaluate impact on emission/absorption line recovery in downstream tasks.

#### Time Series (Light Curves)

- 1D patching (chunk light curve into fixed-length temporal windows) → **temporal shuffle** → MLP projection → SmolLM.

Same principle as wavelength shuffle, applied to the time axis. Adjacent time steps are packed into the feature dimension, compressing long light curves into manageable token sequences.

#### Scalar Catalog Values

- Linear projection → SmolLM.

Scalar properties (magnitudes, redshifts, masses, etc.) are projected directly into the embedding space. Trivial and requires no pretraining.

#### Modality Embeddings

Each token receives a learned modality embedding (image, spectrum, time series, scalar) so the backbone can distinguish data types. All tokens for a given astronomical object are concatenated into a single sequence.

---

## Data: The Multimodal Universe

MMU provides 100TB+ of astronomical data from major surveys, already hosted on Hugging Face. The dataset includes imaging (DESI Legacy Survey, Hyper Suprime-Cam), spectroscopy (SDSS, DESI), photometry (Gaia), and derived physical properties, covering 200M+ observations.

### Prerequisites

Before training can begin, two infrastructure tasks must be completed:

1. **Upload remaining MMU data to Hugging Face.** Not all MMU datasets are currently on HF. The full corpus needs to be available and streamable.
2. **Server-side cross-matching.** Objects observed across multiple surveys need to be linked so that a single training example can include an image from DESI, a spectrum from SDSS, and scalar properties from Gaia for the same object. This cross-matching must work at scale via the MMU Streaming infrastructure.

These are the true bottlenecks. Model architecture and training are straightforward once the data pipeline is in place.

---

## Training Strategy

### Why Continual Pretraining

Three options were considered:

- **Training from scratch** gives full control but discards all general knowledge (reasoning, numerical understanding, sequential structure) that SmolLM already has. It is also the most expensive option, requiring the full compute budget of a from-scratch 3B model. Not practical for a small team when a strong starting point exists.
- **Finetuning** (freezing most of the backbone, training adapters or final layers) is cheap but insufficient. The domain gap between web text and astronomical observations is too large — the model has never seen galaxy image patches or spectral tokens. Finetuning assumes the base representations are mostly correct and need minor adjustment, which is not the case here.
- **Continual pretraining** starts from SmolLM's pretrained weights, which provide a transformer that already understands attention patterns, numerical relationships, and sequential structure. Training continues on MMU data, letting the model gradually adapt to astronomical representations while retaining useful general capabilities. This is the same approach used by AstroLLaMA (continual pretraining of LLaMA on astronomy abstracts) and conceptually mirrors how SmolVLM trains its backbone on multimodal data starting from SmolLM weights.

We adopt continual pretraining from SmolLM3 weights.

### Staged Training

Following SmolVLM's approach, training proceeds in two stages:

**Stage 1 — Connector warmup.** Freeze the SmolLM backbone and SigLIP. Train only the new modality projection layers (image MLP, spectrum MLP, time series MLP, scalar projection) on MMU data. This teaches the connectors to map astronomical data into the backbone's existing embedding space without disrupting its pretrained representations. Relatively cheap and fast.

**Stage 2 — Full continual pretraining.** Unfreeze the SmolLM backbone. Train the full model end-to-end on MMU data (connectors + backbone jointly). SigLIP remains frozen or is lightly tuned with a low learning rate. This is where the backbone adapts its internal representations to astronomical data and learns cross-modal relationships.

This staged approach avoids catastrophic forgetting in early training (when the connectors produce garbage tokens that could corrupt backbone weights) and is the standard recipe for multimodal continual pretraining.

### Preserving Language Capabilities

A benefit of continual pretraining from SmolLM is that the model retains text understanding alongside astronomical modalities. This opens the door to future instruction tuning — for example, "here is a galaxy image and its spectrum, describe what you see" — essentially integrating the AstroLLaVA vision into the foundation model from the start rather than bolting it on later. To preserve language capabilities during Stage 2, a small fraction of text data (e.g., astronomy abstracts from ADS, or a subset of SmolLM's original pretraining mix) can be mixed into the MMU training data.

### General Training Details

- **Optimizer**: AdamW with linear warmup and cosine decay.
- **Scaling**: Train the full model family (135M, 360M, 1.7B, 3B) to characterize scaling laws for astronomical data and enable direct comparison with AstroPT's established scaling results. All sizes are continually pretrained from corresponding SmolLM checkpoints.
- **Context length**: SmolLM3 supports up to 128k tokens. Shuffle ratios for spectra and time series should be set so that a multimodal object (image + spectrum + light curve + scalars) fits comfortably within context.
- **Learning rate**: Stage 2 uses a lower peak learning rate than Stage 1 to avoid catastrophic forgetting of pretrained representations. Follow SmolVLM's learning rate schedule as a starting point.

---

## Evaluation

### Downstream Science Tasks

Evaluate via linear and MLP probes on physical properties, following the framework established in Sanjaripour, Smith et al. (ICLR 2026):

- Photometric properties: magnitudes, colors.
- Spectroscopic properties: redshifts, star formation rates, stellar masses.
- Morphological properties: smoothness, disk fraction, spiral structure.

This physically grounded evaluation avoids the circularity of human-annotated benchmarks and enables direct comparison with both AstroPT and AION-1.

### Direct Comparisons

- **vs. AstroPT (89M, single-modality)**: Does multimodal training improve representations compared to image-only pretraining?
- **vs. AION-1 (3B, bespoke tokenizers)**: At the same parameter count, does the simple SmolVLM-style architecture match AION-1's bespoke multi-tokenizer approach?
- **Tokenization ablations**: Compare affine vs. MLP vs. VQ-VAE at 3B scale to test whether the VQ-VAE advantage observed at 89M (ICLR 2026 paper) persists at scale.
- **Shuffle ratio ablations**: Test 2×, 4×, 8× wavelength/temporal shuffle to characterize the information-compression tradeoff for scientific data.

### The Bitter Lesson Test

The central experiment: does a 3B-parameter GPT with off-the-shelf SigLIP and simple affine/MLP projections, trained on well-curated MMU data, match a 3B-parameter model with 39 bespoke tokenizers trained on a national supercomputer? If yes, it is a strong demonstration that general architectures plus good data suffice for scientific foundation models.

---

## Relationship to Ongoing Work

### Polymathic AI / AION-1

Mike contributes to AION pretraining as a Polymathic collaborator. SmolLM AstroPT is complementary, not adversarial — it explores the opposite end of the complexity spectrum on overlapping data. The two efforts together characterize the design space for astronomical foundation models.

### Multimodal Universe

SmolLM AstroPT is a primary consumer of MMU infrastructure. The data upload and cross-matching work required for this project benefits the entire MMU collaboration and community.

### UniverseTBD

Development will be coordinated via UniverseTBD's Discord, following the open-science model used for AstroPT, AstroLLaMA, and AstroLLaVA. Community contributions welcome.

---

## Key Design Principles

1. **No bespoke tokenizers.** Use pretrained SigLIP for images and minimal projections for everything else. No VQ-VAE, no normalizing flows, no custom encoders per survey.
2. **Follow SmolVLM.** Stay as close to Hugging Face's architecture and training recipes as possible. This maximizes compatibility with existing tooling and community adoption.
3. **Wavelength and temporal shuffle.** Extend SmolVLM's pixel shuffle to 1D scientific data. A simple, principled compression strategy that keeps token counts manageable without discarding information.
4. **Evaluate on physics, not pixels.** Reconstruction quality is insufficient (ICLR 2026). Evaluate representations against independently measured physical quantities.
5. **Scale is the experiment.** The central question is whether simple methods that underperform at 89M catch up or surpass complex methods at 3B. Train the full model family to find out.
6. **Open everything.** Code, weights, data, and training recipes released under permissive licenses. If a result can't be reproduced on a single GPU, it should at least be reproducible on a small cluster.

---

## Timeline and Dependencies

| Phase | Task | Dependency |
|-------|------|------------|
| **0** | Upload remaining MMU data to HF | MMU collaboration |
| **0** | Implement server-side cross-matching in MMU Streaming | HF / LINCC Frameworks |
| **1a** | Implement modality projections and wavelength/temporal shuffle | Phase 0 |
| **1b** | Stage 1: Connector warmup on 135M and 360M (backbone frozen) | Phase 1a |
| **1c** | Stage 2: Full continual pretraining on 135M and 360M | Phase 1b |
| **2a** | Stage 1: Connector warmup on 1.7B and 3B | Phase 1c results |
| **2b** | Stage 2: Full continual pretraining on 1.7B and 3B | Phase 2a |
| **2c** | Run evaluation suite (probes, ablations, comparisons) | Phase 2b models |
| **3** | Paper, model release, community tools | Phase 2c |

Phases 0 and 1a can partially overlap. Phase 0 (data infrastructure) is the critical path. Stage 1 (connector warmup) is relatively cheap and fast at all model sizes; Stage 2 (full continual pretraining) is where the bulk of compute is spent.

---

## Expected Outputs

- **Models**: SmolLM AstroPT family (135M, 360M, 1.7B, 3B), released on Hugging Face.
- **Methods**: Wavelength shuffle and temporal shuffle as named techniques for 1D scientific data tokenization.
- **Results**: Scaling laws for multimodal astronomical foundation models. Direct comparison of simple vs. bespoke tokenization at 3B scale.
- **Infrastructure**: Complete MMU dataset on HF with server-side cross-matching. Benefits the entire field.
- **Paper**: Targeting a top ML venue (NeurIPS, ICML, or ICLR).
