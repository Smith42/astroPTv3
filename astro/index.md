# AstroPTv3

AstroPTv3 is a from-scratch suite of multimodal astronomical foundation
models (70M–12B, Pythia-mirrored) — a SmolLM3 decoder body fed continuous
image/spectrum patch tokens with per-modality regression heads, pretrained
on the Multimodal Universe streamed live from the Hugging Face hub.

```{toctree}
:maxdepth: 2

README
PLAN
EXPERIMENTS
```

```{toctree}
:caption: Guides
:maxdepth: 2

docs/README
docs/training
docs/architecture
docs/jetformer_plan
docs/jetformer_run_guide
docs/jetformer_noise_diagnosis
docs/physical_norm_plan
```

```{toctree}
:caption: Architecture decision records
:maxdepth: 1
:glob:

docs/adr/*
```

```{toctree}
:hidden:
:glob:

docs/evidence/*/*
```
