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
4. appends one JSON line per step to ``{out}/probe_results.jsonl``.

Usage (training machine, alongside a run):
    python astro/scripts/run_probe_sweep.py \
        --checkpoints-dir ../astroPTv3_checkpoints/astropt3-70m \
        --out-dir ../astroPTv3_eval/astropt3-70m \
        --data-root <val_shards_dir> --norm-stats astro/configs/data/pilot_images_spectra.yaml \
        --watch --until-step 143000
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


def process_step(step: int, args) -> dict:
    from astropt3.eval.linear_probe import probe_checkpoint
    from astropt3.eval.val_loss import evaluate_checkpoint

    checkpoint = Path(args.checkpoints_dir) / str(step)
    hf_dir = Path(args.out_dir) / "hf" / str(step)
    convert_checkpoint(Path(args.converter), checkpoint, hf_dir)

    result = {"step": step, "hf_checkpoint": str(hf_dir)}
    val = evaluate_checkpoint(
        hf_dir,
        args.data_root,
        norm_stats=args.norm_stats,
        n_batches=args.val_batches,
        micro_batch_size=args.micro_batch_size,
        seq_len=args.seq_len,
        device=args.device,
        seed=args.seed,
    )
    result["val_loss"] = val["loss"]
    result["val_modality_losses"] = val["modality_losses"]
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
    result["probe_r2"] = probe["r2"]
    result["probe_lambda"] = probe["lambda"]
    result["probe_target"] = probe["target"]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints-dir", required=True, help="nanotron run checkpoint dir")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--data-root", required=True, help="val shard dir or 'synthetic'")
    parser.add_argument("--norm-stats", default=None)
    parser.add_argument("--converter", default=str(DEFAULT_CONVERTER))
    parser.add_argument("--val-batches", type=int, default=512)
    parser.add_argument("--probe-objects", type=int, default=2048)
    parser.add_argument("--target", default="Z")
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--objects-per-batch", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=896)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--watch", action="store_true", help="poll until --until-step is processed")
    parser.add_argument("--poll-interval", type=float, default=60.0)
    parser.add_argument("--until-step", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "probe_results.jsonl"

    while True:
        done = processed_steps(results_path)
        todo = [s for s in completed_steps(Path(args.checkpoints_dir)) if s not in done]
        for step in todo:
            print(f"[sweep] processing step {step}", flush=True)
            result = process_step(step, args)
            with open(results_path, "a") as f:
                f.write(json.dumps(result) + "\n")
            print(f"[sweep] step {step}: val_loss={result['val_loss']:.4f} r2={result['probe_r2']:.4f}", flush=True)
            done.add(step)
        if not args.watch or (args.until_step is not None and args.until_step in done):
            break
        time.sleep(args.poll_interval)

    print(f"[sweep] results in {results_path}", flush=True)


if __name__ == "__main__":
    main()
