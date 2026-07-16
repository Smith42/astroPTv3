"""Sanity-check and benchmark the prepared pilot shards.

Two Phase 2 verification gates, offline (no network):

- decoded-object sanity: stream a few records through the calibrated
  ``ObjectSequencer`` and print per-modality patch statistics (expect ~N(0,1)
  after stretch + per-patch standardization) and the spectrum wavelength
  range (expect 3600-9824 A);
- dataloader-only throughput: records -> ObjectSequencer -> PackedCollator
  inside DataLoader workers, reported as tokens/s. Compare against training
  consumption (tokens/s of the training run) — want >= 2x.

    uv run python scripts/check_pilot_data.py [--data-dir {root}/train] \\
        [--bench-seconds 30] [--workers 4] [--target-tokens-per-sec N]
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from astropt3.config_io import (  # noqa: E402
    load_model_config,
    resolve_data_root,
)
from astropt3.data.mmu import MMUIterableDataset  # noqa: E402
from astropt3.data.packing import ObjectSequencer, PackedCollator  # noqa: E402
from astropt3.tokenization import normalize_wavelength  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "data" / "pilot_images_spectra.yaml"


def sanity(dataset, sequencer, n_objects: int) -> None:
    print(f"=== decoded-object sanity ({n_objects} objects) ===")
    lambda_min, lambda_max = float("inf"), float("-inf")
    for i, record in enumerate(dataset):
        if i >= n_objects:
            break
        if record.get("spectrum") is not None:
            lam_raw = record["spectrum"]["lambda"]
            lambda_min = min(lambda_min, float(lam_raw.min()))
            lambda_max = max(lambda_max, float(lam_raw.max()))
        seq = sequencer.build(record)
        parts = [f"object {seq.object_id}: {len(seq)} tokens"]
        for name, values in seq.values.items():
            patch_means = values.mean(dim=-1)
            patch_stds = values.std(dim=-1)
            parts.append(
                f"  {name}: {len(values)} patches, per-patch mean "
                f"{patch_means.mean():+.4f} (max |{patch_means.abs().max():.4f}|), "
                f"per-patch std {patch_stds.mean():.4f}"
            )
            if name == "spectra":
                lam = seq.positions[name] * 7000.0 + 3000.0  # normalize_wavelength^-1
                parts.append(
                    f"    lambda range [{lam.min():.0f}, {lam.max():.0f}] A, "
                    f"positions [{seq.positions[name].min():.3f}, "
                    f"{seq.positions[name].max():.3f}]"
                )
        print("\n".join(parts))
    expected = normalize_wavelength(torch.tensor([3600.0, 9824.0]))
    print(
        f"(expect image patches ~N(0,1); spectra positions within "
        f"[{expected[0]:.3f}, {expected[1]:.3f}])"
    )
    if lambda_min <= lambda_max:  # at least one spectrum seen
        print(f"spectrum lambda range: [{lambda_min:.1f}, {lambda_max:.1f}] A")
        if lambda_min < 3600.0 or lambda_max > 9824.0:
            print("warning: lambda range outside the expected DESI 3600-9824 A")


class RecordToBatch:
    """Picklable collate_fn: records -> ObjectSeqs -> packed batch."""

    def __init__(self, sequencer: ObjectSequencer, collator: PackedCollator):
        self.sequencer = sequencer
        self.collator = collator

    def __call__(self, records: list[dict]) -> dict:
        batch = self.collator([self.sequencer.build(r) for r in records])
        batch["n_objects"] = len(records)
        return batch


def bench(dataset, sequencer, collator, seconds: float, workers: int, target) -> None:
    print(f"\n=== dataloader throughput ({workers} workers, {seconds:.0f}s) ===")
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=64,
        num_workers=workers,
        collate_fn=RecordToBatch(sequencer, collator),
        prefetch_factor=2 if workers else None,
        persistent_workers=False,
    )
    n_objects = n_rows = n_tokens = 0
    started = time.time()
    for batch in loader:
        rows, seq_len = batch["input_ids"].shape
        n_rows += rows
        n_tokens += rows * seq_len
        n_objects += batch["n_objects"]
        elapsed = time.time() - started
        if elapsed >= seconds:
            break
    elapsed = time.time() - started
    tokens_per_sec = n_tokens / elapsed
    print(
        f"{n_objects} objects -> {n_rows} packed rows in {elapsed:.1f}s: "
        f"{n_objects / elapsed:.1f} obj/s, {tokens_per_sec:,.0f} tokens/s"
    )
    if target is not None:
        verdict = "PASS (>= 2x)" if tokens_per_sec >= 2 * target else "FAIL (< 2x)"
        print(f"vs training consumption {target:,.0f} tokens/s: {verdict}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args_config, _ = parser.parse_known_args()
    data_config = yaml.safe_load(args_config.config.read_text())
    parser.add_argument(
        "--data-dir", type=Path, default=resolve_data_root(data_config) / "train"
    )
    parser.add_argument(
        "--model-config", type=Path, default=ROOT / "configs" / "model" / "test-tiny.yaml"
    )
    parser.add_argument("--sanity-objects", type=int, default=4)
    parser.add_argument("--bench-seconds", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument(
        "--target-tokens-per-sec",
        type=float,
        default=None,
        help="training-side consumption to compare against (want >= 2x)",
    )
    args = parser.parse_args()

    model_config, _ = load_model_config(args.model_config)
    sequencer = ObjectSequencer(model_config)
    seq_len = args.seq_len or data_config["packing"]["seq_len"]
    collator = PackedCollator(model_config, seq_len=seq_len)

    sanity(
        MMUIterableDataset(args.data_dir), sequencer, args.sanity_objects
    )
    if args.bench_seconds > 0:
        bench(
            MMUIterableDataset(args.data_dir, shuffle_buffer_size=256, seed=0),
            sequencer,
            collator,
            args.bench_seconds,
            args.workers,
            args.target_tokens_per_sec,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
