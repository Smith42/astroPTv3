# JetFormer completion plan: noise curriculum, nanotron fork, generation

Status: the `tokeniser: jetformer` option (per-modality `TinyFlow1D` +
`GMMHead`, loss `mean(NLL_GMM(z) − logdet)`) is implemented and tested on the
HF side (`astro-phase5`, see AGENTS.md §4). J1 (noise curriculum), J2
(`astropt3.generation` + `scripts/generate.py`) and J3 (nanotron fork:
flows/GMM heads/loss, TP-synced curriculum noise, conversion,
`test_jetformer_gpu.py`) are implemented; J4 runs on the reserved GH200 node
(GPU venv = `miniforge3_pytorch/2.12.0` module + `.venv-train` overlay —
torch 2.12+cu130 with prebuilt sm_90 flash-attn; the PLAN Phase 3 x86 wheel
recipe does not apply on aarch64). The test pretraining config is
`astro/configs/nanotron/astropt3-70m-jetformer.yaml` against
`/work/nvme/bfvh/msmith10/astroPTv3_data/shakeout_mix2/train`.
Everything is additive — the affine path, the staged Phase 5 pilots, and all
existing gates are untouched.

## Phase J1 — noise curriculum (CPU, this machine)

Port v2 sogol_branch semantics exactly (`JetformerImageEncoder.forward`):

- `sigma = noise_max + (noise_min − noise_max) · frac` with `frac` going
  0 → 1 over training (anneals `noise_max` → `noise_min`).
- Gaussian noise is added to the **flowed z tokens on the input/embedding
  side only**; the GMM targets and the logdet stay clean. Training mode only.

Changes:

- `configuration_astropt3.py`: `jetformer_noise_max: float = 0.1`,
  `jetformer_noise_min: float = 0.0`.
- `modeling_astropt3.py`: `self.jet_noise_frac = 1.0` attribute +
  `set_jet_noise_frac(frac)`. The jetformer branch currently reuses one z for
  embedding and target; split it — noised copy into
  `assemble_inputs_embeds`, clean z into the loss. Default `frac = 1.0`
  (sigma = `noise_min` = 0), so eval/inference and every existing test are
  unaffected.
- `train_smoke.py`: when the config is jetformer, call
  `model.set_jet_noise_frac(step / steps)` each step (~3 lines).
- Tests (`test_jetformer.py`): sigma endpoints (`frac=1` ⇒ deterministic
  eval already covered by pad-invariance; `frac=0`, `model.train()` ⇒
  embeddings vary across calls, loss stays finite); smoke still learns with
  the curriculum on.

## Phase J2 — sampling/generation (CPU-testable now)

New `astro/scripts/generate.py`, HF-side, works on any checkpoint dir
(`import astropt3` + `AutoModel.from_pretrained`):

- **Modes**: `unconditional` (`<|bos|>` → full image span → spectra span),
  `image-to-spectra` (teacher-force a real val record's image tokens,
  generate the spectra span — the scientifically interesting demo), and
  `reconstruct` (one-step teacher-forced predictions; also works for affine
  checkpoints).
- **Sampling** (GIVT recipe): categorical over mixture weights, then a
  Gaussian draw; `--temperature` scales sigma (variance scaling is the GMM
  analogue of temperature); `--argmax` for the mixture-mean point sample.
- **Token loop**: re-run the full forward per generated token. No KV cache
  (`use_cache` stays False). ponytail: O(T²) at ~400 tokens is trivial;
  add caching only if generation ever becomes a hot path.
- **Positions**: images use index positions 0..360; spectra positions come
  from the canonical wavelength grid via a template record through
  `ObjectSequencer`.
- **Output**: inverse flow (`flows[m](z, reverse=True)`) → depatchify →
  inverse asinh (stats from the data config); PNG grid for images,
  flux-vs-wavelength PNG + `.npy` for spectra.
- Test: `tests/test_generate.py` (no gpu marker) — tiny jetformer checkpoint
  from a short smoke run; assert output shapes/finiteness and that two seeds
  give different samples while `--argmax` is deterministic.

## Phase J3 — nanotron fork additions (written here, verified on GPU)

Fork (`nanotron/` submodule, branch `main`), mirroring how affine is wired:

- `config/astropt3_config.py`: accept `"jetformer"` in the tokeniser assert;
  add `jetformer_flow_steps/flow_hidden/gmm_k/noise_max/noise_min`.
- `models/astropt3.py` (duplicate the four small modules from
  `astro/src/astropt3/modalities.py` with **identical attribute names** —
  they are part of the conversion contract):
  - `AstroPT3Embedding` owns `self.flows` when jetformer; its forward flows
    values → z, applies the noise curriculum to the embedded copy, and emits
    extra output keys `{m}_z` [n, D] and `{m}_logdet` [n] alongside
    `input_embeds` (PP=1 is asserted, so dict outputs pass through the
    PipelineBlock locally; declare the new `module_output_keys`).
  - `AstroPT3ModalityHead`: jetformer swaps each `Decoder` for a `GMMHead`.
    `{m}_pred` stays a single tensor — the raw `K·(1+2D)` projection —
    unpacked inside the loss, so the block graph and key set keep their
    shape.
  - `AstroPT3Loss`: jetformer branch reshapes `{m}_pred`, computes
    `gmm_nll({m}_z, …) − {m}_logdet` per token, means, and keeps the same
    `loss_weight` / `n_present` / zero-length-modality semantics (absent
    modality contributes `pred.sum() * 0` to stay in the graph for DDP).
  - Noise frac: `set_jet_noise_frac` on `AstroPT3ForTraining` delegating to
    the embedding block; one guarded line in the fork trainer (next to the
    existing lazy `checkpoint_schedule` hook) sets
    `iteration / total_train_steps` each step.
  - TP: flows and GMM heads are plain `nn.Linear` stacks, so the existing
    `mark_unsharded_params_as_tied_across_tp` + the
    `MODULE_TO_PARAMETRIZE[nn.Linear]` init mapping cover them with **no new
    mechanism** — the TP=2 test verifies, not new code.
- `tools/astropt3/convert_weights.py`: extend both directions —
  HF `flows.{m}.blocks.{i}.net.{0,2}.weight` ↔ fork
  `…token_position_embeddings.pp_block.flows.…`, HF `decoders.{m}.proj.*` ↔
  fork `…modality_head.pp_block.decoders.…`.
- astro side: `nanotron_loader.hf_config_from_modalities` passes the
  jetformer fields through (getattr with defaults, so old fork configs load).
- New `tests/test_jetformer_gpu.py` (`@pytest.mark.gpu`):
  1. HF ↔ nanotron loss parity on a fixed batch after conversion, eval mode
     (noise off), both conversion directions;
  2. TP=2: flow + GMM-head grads bit-identical across ranks;
  3. 50-step CUDA smoke — loss decreases, then convert-to-HF and re-verify;
  4. kill/resume with a jetformer config reproduces the micro-batch stream
     (reuses the Phase 4 harness; the loss path is orthogonal to stream
     state, so this is a cheap regression guard).

## Phase J4 — GPU session: verification + test pretraining run

Needs: leased box with **≥2 GPUs** (TP=2 parity test; everything else runs
on one), A100/H100 class; venv per the PLAN Phase 3 recipe; pilot data via
`ASTROPT3_DATA_ROOT` (rsync or shared FS).

1. **Gate**: full gpu-marked suite (existing affine tests + J3's jetformer
   tests) green.
2. **Test pretraining run**: new `pilot-70m-jetformer` nanotron config —
   identical to the affine 70M shakeout except `tokeniser: jetformer` +
   noise fields, same step budget so the curves are comparable; wandb on
   (smith42).
3. **Acceptance**:
   - loss (NLL − logdet; negative values are correct) decreasing and stable
     — watch grad norms early: nats-scale loss vs Huber's ~0.1 scale is the
     main hyperparameter unknown; knobs if unstable are `noise_max`, lr,
     clip;
   - mid-run kill/resume reproduces the object stream (`object_id_log`);
   - `run_probe_sweep` over the checkpoints: linear-probe redshift R² vs
     the affine 70M shakeout (expectation from the tokenizer benchmark:
     comparable or slightly worse probes — the win is likelihood +
     generation, not probing);
   - `generate.py` sample grid + image→spectra from the final checkpoint,
     eyeballed for coherence.
4. **Deliverables**: wandb links, probe comparison table, sample grids, and
   a go/no-go note on adding jetformer to the suite's ablation matrix.

## Sequencing and risks

- J1 + J2 are a few hours on this machine and have no GPU dependency; J3
  code is written here too, so the GPU session starts at "run tests", not
  "write code". Lease the GPU once J1–J3 are pushed.
- Main technical risk is flow/NLL training stability at real-data scale
  (the curriculum exists precisely for this). Second risk is loss-scale
  interaction with clipping/lr; both are observable in the first ~500 steps
  of the J4 run, which is why the test run precedes any ablation-matrix
  decision.
- Conversion must land before the pretraining run (the probe sweep converts
  every checkpoint to HF).
- Fork changes go on the fork's `main` as before; bump the submodule
  pointer in this repo in the same commit as the astro-side changes.
