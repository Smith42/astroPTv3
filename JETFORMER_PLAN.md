# JetFormer — status handoff & run guide

Handoff note for picking this work up in a fresh session. It covers (1) what
the first 70M test run showed, (2) how to launch a JetFormer training run,
(3) the state of reconstruction logging, and (4) the agreed next task
(in-trainer reconstruction figures). The **implementation** plan (how the
JetFormer tokeniser / fork were built, J1–J4) lives in
`astro/docs/jetformer_plan.md` and is not repeated here.

Model = SmolLM3 body + per-modality `TinyFlow1D` (RealNVP affine couplings) +
`GMMHead`; loss is the exact patch-space likelihood `mean(NLL_GMM(z) − logdet)`
(can and should go **negative**). Per-patch standardization is skipped for
jetformer so the record→token map stays invertible.

---

## 1. What the first test run showed (70M, wandb `17k4i9n1`)

Run `astropt3-70m-jetformer` on 2×GH200 (DP=2, TP=1, PP=1), 20,000 steps,
2.62B tokens, bf16. Completed cleanly — all 30 Pythia-schedule checkpoints
(1,2,4,…,512 then every 1000) at
`/work/nvme/bfvh/msmith10/astroPTv3_checkpoints/astropt3-70m-jetformer/`,
`latest.txt` = 20000.

**Good signals**
- The exact-likelihood objective works: image NLL fell from **+799 → ≈ −38**
  (negative is correct for a continuous density). Crosses zero ~step 5000,
  then a noisy −25…−40 plateau.
- **Reconstruction fidelity is real.** One-step teacher-forced reconstructions
  of held-out val galaxies correlate **0.69–0.90** with the true pixels
  (MSE 0.12–0.21 vs. truth variance 0.25–0.61). The model learned genuine
  image structure.

**Red flags (address before scaling to the ablation matrix)**
- **Gradient norms explode over training.** A few hundred early → 1–2K
  mid-run → **9K, 23K, 15K** in the last quarter, all against `clip_grad: 1.0`.
  Every update is clip-limited by a factor rising to ~15,000–40,000×; the clip
  is doing all the step-size control. This is the nats-scale-loss vs.
  raw-target interaction the plan flagged, and it grows as the dequantization
  noise anneals to `noise_min: 0.0` (nothing then floors the flow from
  over-sharpening the density). Knobs to try: a **nonzero `noise_min`**, a
  **lower LR**, and/or a **larger `clip_grad`**. This is the main go/no-go
  question.
- **Spectra effectively did not train, and there are no real spectra in the
  data.** `shakeout_mix2` has a `spectrum` struct column but it is **null in
  every val and train cone shard checked (0 of ~5,400 rows)** — the crossmatch
  kept the redshift labels (`Z`, `z_spec`) but not the spectrum arrays. The
  nonzero `spectra_loss` during training (~17% of steps) came from the
  **synthetic** records mixed in via `synthetic_image_only_fraction: 0.3`.
  Spectra NLL plateaued (~260–280) and never really improved. If spectra
  matter, a dataset that actually carries spectrum arrays is required first.

---

## 2. How to run a JetFormer training run

### Environment (DeltaAI GH200, aarch64) — required every fresh shell
No module system or GPU is visible on the login node; the training happens in
a Slurm GPU step. The venv is a `--system-site-packages` overlay on the module
torch (aarch64 flash-attn ships prebuilt in the module — do **not** rebuild).

```bash
source /etc/profile.d/modules.sh
module use /sw/user/modules/python
module load python/miniforge3_pytorch/2.12.0   # torch 2.12+cu130, flash-attn (sm_90), transformers 4.57
source .venv-train/bin/activate                # at repo root; create once (see below)
```

First-time venv creation (once):
```bash
python -m venv --system-site-packages .venv-train
.venv-train/bin/pip install "datasets>=4.3" torchdata pytest -e ./nanotron -e ./astro
```

### Launch (from the repo root)
```bash
nvidia-smi -L                      # confirm BOTH GPUs are visible; config wants dp: 2
bash astro/scripts/launch_jetformer_70m.sh
```
The launcher `module load`s, activates `.venv-train`, sets
`CUDA_DEVICE_MAX_CONNECTIONS=1` / `HF_DATASETS_OFFLINE=1` / `WANDB_MODE=online`,
and runs `python -m torch.distributed.run --nproc-per-node=<#GPUs>
nanotron/run_train.py --config-file astro/configs/nanotron/astropt3-70m-jetformer.yaml`.
Extra torchrun/nanotron args pass straight through.

### Config: `astro/configs/nanotron/astropt3-70m-jetformer.yaml`
Key fields (and the knobs most likely to change):
- data: `data_root: .../shakeout_mix2/train`, `norm_stats:
  astro/configs/data/pilot_images_spectra.yaml`, `synthetic_image_only_fraction:
  0.3`, `object_id_log: <ckpt>/objects.log`, `num_loading_workers: 8`.
- jetformer: `tokeniser: jetformer`, `jetformer_flow_steps: 4`,
  `jetformer_flow_hidden: 128`, `jetformer_gmm_k: 4`,
  **`jetformer_noise_max: 0.1`, `jetformer_noise_min: 0.0`** (raise the min to
  fight the grad-norm growth).
- optim: **`clip_grad: 1.0`**, `learning_rate: 1e-3` cosine → `min_decay_lr:
  1e-4`, `lr_warmup_steps: 200`, `weight_decay: 0.1`, adamW (β 0.9/0.95).
- tokens: `micro_batch_size: 16`, `sequence_length: 4096`, `train_steps: 20000`.
- parallelism: `dp: 2, tp: 1, pp: 1, tp_mode: ALL_REDUCE` (PP=1 and ALL_REDUCE
  are asserted — modality modules are TP-replicated).
- checkpoints: `checkpoint_schedule: pythia`, `checkpoint_interval: 1000`.

A tiny CPU-smoke config exists at
`astro/configs/nanotron/astropt3-test-tiny-jetformer.yaml`.

### While it runs
- Watch on wandb (project `smith42/astropt3`): `lm_loss` (negative = healthy),
  `images_loss`, `spectra_loss`, and **`grad_norm`** (the thing to watch — see
  §1). Throughput ≈ 400K tok/s, ~200 TFLOPs/GPU, ~310 ms/step on 2×GH200.
- **Kill/resume** is exact: `resume_checkpoint_path` (or the trainer's
  auto-resume) re-draws the untrained partial packing row and continues the
  micro-batch stream; resume with the **same `num_loading_workers`**.

### Evaluation (never in the trainer — separate process, spare GPU)
```bash
python astro/scripts/run_probe_sweep.py \
  --checkpoints-dir /work/nvme/.../astropt3_checkpoints/astropt3-70m-jetformer \
  --out-dir /work/nvme/.../astroPTv3_eval/astropt3-70m-jetformer \
  --data-root <val_shards_dir> --norm-stats astro/configs/data/pilot_images_spectra.yaml \
  --watch --until-step 20000
```
Polls `latest.txt`, converts each step to HF, runs `val_loss` + redshift
`linear_probe`, appends `probe_results.jsonl`. Converting a checkpoint to HF
needs a GPU (the converter imports flash-attn); on a GPU-less login node use
the direct-safetensors CPU path (see the `cpu-nanotron-to-hf-conversion`
memory / `scratchpad/convert_cpu.py`).

---

## 3. Reconstruction logging — current state

`astro/scripts/generate.py` renders reconstructions/samples from an HF
checkpoint and can log them to wandb. This session added multi-record batching
(`--record-index 0,1,2,3` → one run, object-id in each key) so a batch lands in
a single run; `--wandb` starts a fresh generation run, `--wandb-run-id <id>`
appends to an existing one. **This change to `generate.py` is uncommitted.**

Limitation that motivated the next task: `generate.py` is **post-hoc** — it
runs against a saved checkpoint and can only append to a run after the fact.
Demo figures from `17k4i9n1` (step-20000) were logged to that run and to a
standalone run `o3fvwjgd`.

---

## 4. Next task — reconstruction figures logged *inside* the training run

**Goal:** log a reconstruction to the **live** wandb run every N steps, next to
the losses, so you can watch quality track training — not post-hoc.

**Key point (why this is nearly free):** the loss forward already computes
everything a teacher-forced reconstruction needs, so **no second transformer
forward** is required.
- `AstroPT3Embedding.forward` (nanotron `models/astropt3.py`) stashes the
  **clean** `{m}_z` (flow-forward of the true patches; the noise only touches a
  separate embedded copy) and `{m}_logdet`.
- `AstroPT3ModalityHead.forward` produces `{m}_pred` = GMM mixture params at the
  predicting positions.

**Recipe** (per modality, on the DP=0 rank, at viz steps):
1. mixture mean in z-space: `z_pred = Σ softmax(logits_pi)_k · mu_k`
   (`unpack_gmm_params({m}_pred, k, D)` gives `logits_pi, mu, log_sigma`);
2. invert the flow: `values_pred = flows[m](z_pred, reverse=True)`
   (`TinyFlow1D.forward(..., reverse=True)` is the inverse — already exists);
3. depatchify `values_pred` (prediction) and `modality_values[m]` (truth) →
   side-by-side figure → `wandb.Image`.

**Design (agreed):**
- Put the reconstruction/render helper in `astro/` so the fork stays thin; the
  trainer just calls it.
- Trainer hook at a configurable interval (like the loss cadence); DP=0 rank
  only (GMM/flow are TP-replicated, so one rank has full weights — no gather).
- Wrap the whole viz block in `try/except` so a matplotlib/wandb/NaN hiccup can
  never kill an expensive run.
- Reconstruct with **noise off** (or accept that reusing the in-flight
  `{m}_pred` reflects training-time noise — minor once the curriculum has
  annealed).
- **Open choice:** reconstruct the *current* training-batch object (truly free,
  but a rotating subject) vs. a *fixed* val galaxy (one small extra forward,
  ~400 tokens, <0.1% overhead, but you can watch one subject sharpen).
  Recommendation: fixed subject — trivial cost, far more readable as a progress
  plot.
- This **overrides the repo's "evaluation never runs in the trainer" rule** on
  purpose. It is justified — reconstruction is cheap and needs no HF conversion,
  unlike the val/probe sweep the rule targets — but comment it clearly so nobody
  later assumes the probe sweep can move inline too.

Plumbing: `{m}_pred` / `{m}_z` / `modality_values` are currently consumed inside
the loss module; only the scalar sub-losses bubble up to the trainer. At viz
steps, surface a small detached slice for one object to the DP=0 rank (mirror
how `{m}_loss` already flows), or compute the figure where those tensors are in
scope and log from there.

---

## 5. Open items for the restart

- **Tune stability before the ablation matrix:** sweep `noise_min` (try
  0.02–0.05), LR (try 5e-4), and `clip_grad` — the grad-norm growth in §1 is the
  gating question. First ~500–2000 steps show whether it's fixed.
- **Real spectra:** `shakeout_mix2` has none; get a dataset with populated
  `spectrum` arrays if the spectra modality is in scope.
- **Commit** the `generate.py` change (and the launcher edit) on the `jetformer`
  branch — pushes happen from the training node, not the login node.
- The HF checkpoint converted this session lives in the session scratchpad
  (temporary); re-convert from the nanotron checkpoint when needed.
- Cross-refs: `astro/docs/jetformer_plan.md` (implementation plan),
  memories `deltaai-gh200-train-env` and `cpu-nanotron-to-hf-conversion`.
