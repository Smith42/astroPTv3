"""Validation loss of a converted HF checkpoint on fixed val batches.

The batch stream is deterministic (rank 0, no shuffle, fixed seed), so every
checkpoint of a run is scored on the SAME ``--batches`` micro-batches and the
numbers are comparable across steps.

Usage:
    python -m astropt3.eval.val_loss --checkpoint <hf_dir> \
        --data-root <val_shards_dir|synthetic> \
        --batches 512 --micro-batch-size 4 --seq-len 896 [--out val.json]
"""

import argparse
import json
from itertools import islice

import torch

from ..data.nanotron_loader import PackedMicroBatches, regroup_micro_batch

# synthetic val stream starts here so it never overlaps training indices
SYNTHETIC_VAL_OFFSET = 10_000_000


def val_batches(config, data_root, *, n_batches, micro_batch_size, seq_len, seed=0):
    """Yield the fixed validation micro-batches (HF forward kwargs)."""
    stream = PackedMicroBatches(
        config,
        micro_batch_size,
        seq_len,
        data_root=data_root,
        rank=0,
        world_size=1,
        seed=seed,
        # ADR 0006 §5: the MMU path draws the reserved val partitions, which
        # are spatially disjoint from train (the synthetic path holds out by
        # record index instead, below)
        split="val",
    )
    if data_root == "synthetic":
        # start the val stream far past any training index (held-out records)
        stream.load_state_dict(
            {
                "records": SYNTHETIC_VAL_OFFSET,
                "epoch": 0,
                "stream_state": None,
                "data_root": "synthetic",
                "source_assembly": "synthetic",
            }
        )
    names = config.modality_registry().names()
    for flat in islice(iter(stream), n_batches):
        yield regroup_micro_batch(flat, names)


@torch.no_grad()
def evaluate(
    model,
    data_root,
    *,
    n_batches=512,
    micro_batch_size=4,
    seq_len=896,
    seed=0,
    batches=None,
):
    """Mean loss (and per-modality means) over the fixed val batches.

    ``batches`` is an optional pre-built list from :func:`val_batches` — the
    batches depend only on the data stream, not on checkpoint weights, so
    sweeps should build them once and reuse them for every step instead of
    re-streaming and re-packing the val shards per checkpoint.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    total, per_key, per_key_n = 0.0, {}, {}
    n = 0
    for kwargs in (
        batches
        if batches is not None
        else val_batches(
            model.config,
            data_root,
            n_batches=n_batches,
            micro_batch_size=micro_batch_size,
            seq_len=seq_len,
            seed=seed,
        )
    ):
        kwargs = {
            k: (
                {
                    kk: vv.to(
                        device=device, dtype=dtype if vv.is_floating_point() else None
                    )
                    for kk, vv in v.items()
                }
                if isinstance(v, dict)
                else v.to(device)
            )
            for k, v in kwargs.items()
        }
        out = model(**kwargs)
        total += out.loss.item()
        for key, value in out.modality_losses.items():
            per_key[key] = per_key.get(key, 0.0) + value.item()
            per_key_n[key] = per_key_n.get(key, 0) + 1
        n += 1
    if n == 0:
        raise ValueError("no validation batches produced")
    return {
        "loss": total / n,
        "modality_losses": {k: per_key[k] / per_key_n[k] for k in sorted(per_key)},
        "n_batches": n,
    }


def evaluate_checkpoint(
    checkpoint,
    data_root,
    *,
    n_batches=512,
    micro_batch_size=4,
    seq_len=896,
    device=None,
    seed=0,
    batches=None,
):
    import astropt3  # noqa: F401  -- registers the Auto classes

    from transformers import AutoModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = AutoModel.from_pretrained(checkpoint).to(device=device, dtype=dtype).eval()
    result = evaluate(
        model,
        data_root,
        n_batches=n_batches,
        micro_batch_size=micro_batch_size,
        seq_len=seq_len,
        seed=seed,
        batches=batches,
    )
    result["checkpoint"] = str(checkpoint)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", required=True, help="converted HF checkpoint dir"
    )
    parser.add_argument(
        "--data-root", required=True, help="val shard dir or 'synthetic'"
    )
    parser.add_argument("--batches", type=int, default=512)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=896)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="optional JSON output path")
    args = parser.parse_args()

    result = evaluate_checkpoint(
        args.checkpoint,
        args.data_root,
        n_batches=args.batches,
        micro_batch_size=args.micro_batch_size,
        seq_len=args.seq_len,
        device=args.device,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))
    if args.out:
        try:
            with open(args.out, "w") as file:
                json.dump(result, file, indent=2)
        except OSError as error:
            raise RuntimeError(f"cannot write validation output {args.out}") from error


if __name__ == "__main__":
    main()
