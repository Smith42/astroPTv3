# AGENTS.md

This file provides guidance to Agents when working with code in this repository.

## What this repo is

AstroPTv3: a from-scratch suite of multimodal astronomical foundation models
(70M–12B, Pythia-mirrored) — a SmolLM3 decoder body fed continuous image/
spectra patch tokens with per-modality regression heads, pretrained on the
Multimodal Universe. The repo is a fork of `huggingface/smollm`: `text/`,
`vision/`, `tools/` are **read-only upstream reference**; all project code
lives in `astro/`, and `nanotron/` is a git submodule pointing at the
pretraining fork (`Smith42/nanotron`, branch `astropt3`).

**Before working in `astro/`, read [`astro/AGENTS.md`](astro/AGENTS.md)** —
it carries the commands, hard constraints, and the architecture contract
for the project. The approved phase plan (decisions are fixed) is
[`astro/PLAN.md`](astro/PLAN.md); the project README is
[`astro/README.md`](astro/README.md).

Repo-wide constraints: GPU work and training runs are allowed on this box
(2×A100 80GB and slurm), but the CPU suite stays the fast gate and
multi-day runs belong on the training cluster by preference; dependencies
are managed only through uv in `astro/pyproject.toml`.
