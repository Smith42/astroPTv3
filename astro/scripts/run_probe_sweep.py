"""Async eval sweep over nanotron checkpoints: convert -> val loss -> probe.

Runs OUTSIDE the trainer (separate process, ideally a spare GPU via
``CUDA_VISIBLE_DEVICES``) so evaluation never blocks training. It polls the
run's checkpoint directory, and for every completed checkpoint step (as
gated by ``latest.txt``, written last during a save):

1. converts it to a HF checkpoint under ``{out}/hf/{step}`` with the fork's
   ``tools/astropt3/convert_nanotron_to_hf.py`` (torchrun subprocess);
2. computes the fixed-batch validation loss (``astropt3.eval.val_loss``);
3. ridge-probes redshift from pooled hidden states
   (``astropt3.eval.linear_probe``);
4. samples/renders fixed-template image + spectra panels
   (``astropt3.eval.samples`` — ADR 0003) into ``{out}/samples/{step}``,
   unless ``--sample-records none``;
5. appends one JSON line per step to ``{out}/probe_results.jsonl``.

``--wandb`` mirrors the scalars and sample panels to the sweep's OWN wandb
run (project astropt3, job_type eval; NOT the trainer's run — the sweep lags
training, which wandb's monotonic internal step won't tolerate). Panels use
``checkpoint_step`` as their x-axis; the run id defaults to
``eval-{checkpoints-dir name}`` so a restarted sweep resumes the same run.
The JSONL stays the authoritative done-step record — steps already in it
before samples/wandb existed are never revisited (use a fresh ``--out-dir``
or delete their lines to re-log; HF conversions are cached, so re-eval is
cheap).

By default every checkpoint the trainer writes is evaluated exactly once
(steps 1, 2, 4, ..., 512 then every ``checkpoint_interval``).
``--eval-every N`` thins that to exact multiples of N -- dropping the early
powers of two, which the Pythia schedule keeps -- and ``--samples-every`` /
``--samples-floor`` set the cadence for the whole evaluation (val loss and
probe included, not just the panels): ``should_checkpoint`` multiples at or
above the floor. ``--until-step``
bounds the sweep: steps above it are never evaluated, and with ``--watch``
polling stops once the trainer reaches it.

Usage (training machine, alongside a run):
    python astro/scripts/run_probe_sweep.py \
        --checkpoints-dir ../astroPTv3_checkpoints/astropt3-70m \
        --out-dir ../astroPTv3_eval/astropt3-70m \
        --data-root <val_shards_dir> \
        --watch --until-step 143000 --wandb

Needs the ``[train]`` extra (matplotlib + wandb for the panels).
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ASTRO_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ASTRO_DIR.parent
DEFAULT_CONVERTER = REPO_ROOT / "nanotron" / "tools" / "astropt3" / "convert_nanotron_to_hf.py"


def completed_steps(checkpoints_dir: Path) -> list[int]:
    """Step dirs covered by latest.txt (a save writes latest.txt last)."""
    latest_file = checkpoints_dir / "latest.txt"
    if not latest_file.exists():
        return []
    latest = int(latest_file.read_text().strip())
    steps = [
        int(p.name)
        for p in checkpoints_dir.iterdir()
        if p.is_dir() and p.name.isdigit() and (p / "model_config.json").exists()
    ]
    return sorted(s for s in steps if s <= latest)


def processed_steps(results_path: Path) -> set[int]:
    if not results_path.exists():
        return set()
    return {json.loads(line)["step"] for line in results_path.read_text().splitlines() if line.strip()}


def steps_to_eval(
    completed: list[int],
    done: set[int],
    until_step: int | None = None,
    eval_every: int = 1,
    samples_every: int = 1,
    samples_floor: int = 0,
) -> list[int]:
    """Completed checkpoint steps still needing evaluation, ascending.

    ``eval_every`` keeps only exact multiples -- unlike the Pythia
    ``should_checkpoint`` schedule that writes the checkpoints, it does NOT
    keep the early powers of two, so ``--eval-every 1000`` on a run with
    ``checkpoint_interval: 1000`` skips steps 1..512 entirely.
    ``samples_every``/``samples_floor`` gate the WHOLE step -- val loss and
    probe included, not just the panels: keep ``should_checkpoint``
    multiples at or above the floor.
    """
    from astropt3.checkpoint_schedule import should_checkpoint

    return [
        s
        for s in completed
        if s not in done
        and (until_step is None or s <= until_step)
        and (eval_every == 1 or s % eval_every == 0)
        and s >= samples_floor
        and should_checkpoint(s, samples_every)
    ]


def convert_checkpoint(converter: Path, checkpoint: Path, save_path: Path) -> None:
    if (save_path / "config.json").exists():
        return  # already converted (e.g. by a previous sweep pass)
    port = random.randint(29600, 29999)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=1",
            f"--master_port={port}",
            str(converter),
            f"--checkpoint_path={checkpoint}",
            f"--save_path={save_path}",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "CUDA_DEVICE_MAX_CONNECTIONS": "1"},
        check=True,
        timeout=3600,
    )


def process_step(step: int, args, sample_records: list[dict]) -> dict:
    from astropt3.eval.linear_probe import probe_checkpoint
    from astropt3.eval.samples import sample_checkpoint
    from astropt3.eval.val_loss import evaluate_checkpoint

    checkpoint = Path(args.checkpoints_dir) / str(step)
    hf_dir = Path(args.out_dir) / "hf" / str(step)
    convert_checkpoint(Path(args.converter), checkpoint, hf_dir)

    result = {"step": step, "hf_checkpoint": str(hf_dir)}
    val = evaluate_checkpoint(
        hf_dir,
        args.data_root,
        n_batches=args.val_batches,
        micro_batch_size=args.micro_batch_size,
        seq_len=args.seq_len,
        device=args.device,
        seed=args.seed,
    )
    result["val_loss"] = val["loss"]
    result["val_modality_losses"] = val["modality_losses"]
    try:
        probe = probe_checkpoint(
            hf_dir,
            args.data_root,
            target=args.target,
            n_objects=args.probe_objects,
            seq_len=args.seq_len,
            objects_per_batch=args.objects_per_batch,
            device=args.device,
            seed=args.seed,
        )
    except ValueError as exc:
        # a val corpus can carry too few labelled objects (shakeout_mix2 has no
        # DESI matches at all); val loss + panels are still worth having
        print(f"[sweep] probe skipped: {exc}", flush=True)
        probe = {"r2": None, "lambda": None, "target": args.target}
    result["probe_r2"] = probe["r2"]
    result["probe_lambda"] = probe["lambda"]
    result["probe_target"] = probe["target"]
    # steps_to_eval already applied the samples cadence: every scheduled step samples
    if sample_records:
        result["samples"] = sample_checkpoint(
            hf_dir,
            sample_records,
            modes=None if args.sample_modes == "auto" else args.sample_modes.split(","),
            n=args.sample_n,
            temperature=args.sample_temperature,
            seed=args.seed,
            out_dir=Path(args.out_dir) / "samples" / str(step),
            device=args.device,
            step=step,
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", required=True, help="nanotron run checkpoint dir")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--data-root", required=True, help="val shard dir or 'synthetic'")
    parser.add_argument("--converter", default=str(DEFAULT_CONVERTER))
    parser.add_argument("--val-batches", type=int, default=512)
    parser.add_argument("--probe-objects", type=int, default=2048)
    parser.add_argument("--target", default="Z")
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--objects-per-batch", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=896)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="only evaluate checkpoint steps that are exact multiples of N "
        "(default 1: every checkpoint). Unlike the Pythia checkpoint schedule "
        "this does NOT keep the early powers of two, so --eval-every 1000 skips "
        "steps 1..512",
    )
    parser.add_argument("--watch", action="store_true", help="poll until --until-step is processed")
    parser.add_argument("--poll-interval", type=float, default=60.0)
    parser.add_argument(
        "--until-step",
        type=int,
        default=None,
        help="highest checkpoint step to evaluate; steps above it are never processed "
        "(pass the run's train_steps to sweep the whole run)",
    )
    parser.add_argument(
        "--sample-records",
        default="0",
        help="comma-separated template record indices for sample panels; an "
        "'s' prefix (e.g. 's0') selects a spectrum-only template (ADR 0005); "
        "'none' disables",
    )
    parser.add_argument(
        "--sample-modes",
        default="auto",
        help="'auto' (jetformer: unconditional+image-to-spectra, affine: reconstruct) "
        "or a comma list of unconditional,image-to-spectra,reconstruct",
    )
    parser.add_argument("--sample-n", type=int, default=4, help="samples per mode")
    parser.add_argument("--sample-temperature", type=float, default=1.0)
    parser.add_argument(
        "--samples-floor",
        type=int,
        default=0,
        help="never evaluate (val loss, probe, or panels) before this step (the "
        "Pythia schedule's early pow2<=512 checkpoints are kept by default; set "
        "e.g. 1000 to suppress them)",
    )
    parser.add_argument(
        "--samples-every",
        type=int,
        default=1,
        help="evaluate when should_checkpoint(step, N): every pow2<=512 plus every Nth step",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="mirror scalars + sample panels to the sweep's own wandb run (project astropt3)",
    )
    parser.add_argument(
        "--wandb-run-id",
        default=None,
        help="wandb run id to resume across sweep restarts (default: eval-{checkpoints-dir name})",
    )
    args = parser.parse_args()
    if args.samples_every < 1:
        parser.error("--samples-every must be >= 1 (disable sampling with --sample-records none)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "probe_results.jsonl"

    # template records are fixed for the whole sweep (and, with a fixed seed,
    # across sweeps of the same run), loaded once: streaming real shards for a
    # spectrum-bearing record per step would be pure waste
    sample_records = []
    if args.sample_records != "none":
        from astropt3.eval.samples import load_template_record

        sample_records = [
            load_template_record(
                args.data_root,
                int(tok.lstrip("s")),
                prefer_spectrum=True,
                spectrum_only=tok.startswith("s"),
            )
            for tok in args.sample_records.split(",")
        ]

    wandb_run = None
    if args.wandb:
        import wandb

        run_id = args.wandb_run_id or f"eval-{Path(args.checkpoints_dir).name}"
        wandb_run = wandb.init(
            project="astropt3",
            id=run_id,
            resume="allow",
            name=run_id,
            job_type="eval",
            config={k: v for k, v in vars(args).items() if k not in ("wandb", "wandb_run_id")},
        )
        # panels plot against the checkpoint step, not wandb's internal step
        # (which just counts log calls in this lagging sidecar run)
        wandb_run.define_metric("checkpoint_step")
        wandb_run.define_metric("*", step_metric="checkpoint_step")

    while True:
        done = processed_steps(results_path)
        completed = completed_steps(Path(args.checkpoints_dir))
        todo = steps_to_eval(
            completed,
            done,
            args.until_step,
            args.eval_every,
            args.samples_every,
            args.samples_floor,
        )
        for step in todo:
            print(f"[sweep] processing step {step}", flush=True)
            result = process_step(step, args, sample_records)
            with open(results_path, "a") as f:
                f.write(json.dumps(result) + "\n")
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "checkpoint_step": step,
                        "val/loss": result["val_loss"],
                        **{f"val/loss_{m}": v for m, v in result["val_modality_losses"].items()},
                        **({} if result["probe_r2"] is None else {"probe/r2": result["probe_r2"]}),
                        **{
                            f"samples/{k}": wandb.Image(p)
                            for k, p in result.get("samples", {}).items()
                        },
                    }
                )
            r2 = result["probe_r2"]
            print(
                f"[sweep] step {step}: val_loss={result['val_loss']:.4f} "
                f"r2={'n/a' if r2 is None else format(r2, '.4f')}",
                flush=True,
            )
            done.add(step)
        if not args.watch:
            break
        # once the trainer is at/past --until-step, every in-range step is in
        # `done` (they were all in this pass's todo), so the sweep is finished.
        # Keyed on the trainer's progress, not on --until-step itself being a
        # scheduled step, so an off-schedule bound still terminates.
        if args.until_step is not None and max(completed, default=0) >= args.until_step:
            break
        time.sleep(args.poll_interval)

    if wandb_run is not None:
        wandb_run.finish()
    print(f"[sweep] results in {results_path}", flush=True)


if __name__ == "__main__":
    main()
