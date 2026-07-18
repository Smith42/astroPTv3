"""Autoregressive scalar prediction metrics (ADR 0008's success gate).

Conditions the model on an object's observation spans (images/spectra and
the non-target scalars), forces the target scalar's span LAST, and reads
the GMM head at the ``starts-1`` position — the model is *asked* for the
value instead of probed for it. Reported for ``Z`` as the literature's
photometric-redshift numbers, computed in the normalized ``log(1+z)``
space where sigma reads directly as ``dz/(1+z)``:

- ``nmad``:          1.4826 * median(|residual|) — robust scatter
- ``outlier_frac``:  fraction with |residual| > 0.15
- ``coverage_1sig``: fraction of truths inside the head's 1-sigma highest-
                     weight component interval (~0.68 when calibrated)

Deterministic for a given checkpoint (teacher-forced, no sampling).

Usage:
    python -m astropt3.eval.scalar_head --checkpoint <hf_dir> \
        --data-root <val_shards_dir|synthetic> --target Z --n-objects 2048
"""

import argparse
import json
import warnings

import numpy as np
import torch

from ..data.packing import ObjectSequencer, PackedCollator
from .linear_probe import _val_records

OUTLIER_THRESHOLD = 0.15  # |d log(1+z)| — the standard photo-z outlier cut


def collect_scalar_objects(config, data_root, target, n_objects, *, seed=0):
    """First ``n_objects`` val objects carrying the target span plus at least
    one other span to condition on; the target span is pinned LAST."""
    sequencer = ObjectSequencer(config)
    objects, targets = [], []
    for record in _val_records(data_root, seed=seed):
        if sequencer._scalar_value(target, record) is None:
            continue
        # everything else the record carries, in registry order, target last
        probe = sequencer.build(record)
        others = [m for m in sorted(probe.masks) if m != target]
        if not others:
            continue
        obj = sequencer.build(record, modality_order=others + [target])
        objects.append(obj)
        targets.append(float(obj.values[target][0, 0]))  # normalized truth
        if len(objects) >= n_objects:
            break
    if not objects:
        raise ValueError(f"no records carry a conditionable {target!r} span")
    if len(objects) < n_objects:
        warnings.warn(
            f"val stream exhausted: scoring {len(objects)}/{n_objects} objects",
            stacklevel=2,
        )
    return objects, np.asarray(targets, dtype=np.float64)


@torch.no_grad()
def scalar_head_metrics(
    model, objects, targets, *, target="Z", seq_len=896, objects_per_batch=8
):
    """Teacher-forced GMM read-out at the target span; metrics in normalized space."""
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    collator = PackedCollator(model.config, seq_len=seq_len)
    preds, sigmas = [], []
    for i in range(0, len(objects), objects_per_batch):
        batch = collator(objects[i : i + objects_per_batch])
        kwargs = {
            k: (
                {kk: vv.to(device=device, dtype=dtype if vv.is_floating_point() else None) for kk, vv in v.items()}
                if isinstance(v, dict)
                else v.to(device)
            )
            for k, v in batch.items()
        }
        out = model(**kwargs, compute_loss=False)
        from ..modeling_astropt3 import left_shift_mask

        mask = kwargs["modality_masks"][target]
        logits_pi, mu, log_sigma = model.decoders[target](
            out.last_hidden_state[left_shift_mask(mask)]
        )
        pi = torch.softmax(logits_pi, dim=-1)
        # mixture mean as the point estimate; the top-weight component's
        # sigma as the reported uncertainty (a full-mixture interval is a
        # refinement the calibration gate does not need yet)
        preds.append((pi.unsqueeze(-1) * mu).sum(dim=-2)[:, 0].float().cpu().numpy())
        top = pi.argmax(dim=-1)
        idx = top.view(-1, 1, 1).expand(-1, 1, mu.size(-1))
        sigmas.append(log_sigma.gather(-2, idx)[:, 0, 0].exp().float().cpu().numpy())
    preds = np.concatenate(preds).astype(np.float64)
    sigmas = np.concatenate(sigmas).astype(np.float64)
    if len(preds) != len(targets):
        raise RuntimeError(f"{len(preds)} predictions for {len(targets)} targets")

    residuals = preds - targets
    return {
        "target": target,
        "n_objects": int(len(targets)),
        "nmad": float(1.4826 * np.median(np.abs(residuals))),
        "outlier_frac": float((np.abs(residuals) > OUTLIER_THRESHOLD).mean()),
        "coverage_1sig": float((np.abs(residuals) <= sigmas).mean()),
        "bias": float(np.median(residuals)),
    }


def evaluate_checkpoint(
    checkpoint,
    data_root,
    *,
    target="Z",
    n_objects=2048,
    seq_len=896,
    objects_per_batch=8,
    device=None,
    seed=0,
    scalar_set=None,
):
    """``scalar_set`` is an optional pre-collected (objects, targets) pair —
    the set depends only on the data stream, so sweeps collect once."""
    import astropt3  # noqa: F401  -- registers the Auto classes

    from transformers import AutoModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model = AutoModel.from_pretrained(checkpoint).to(device=device, dtype=dtype).eval()
    if target not in model.config.modality_registry().names():
        raise ValueError(f"checkpoint has no {target!r} modality (pre-0008 config)")
    if scalar_set is None:
        scalar_set = collect_scalar_objects(
            model.config, data_root, target, n_objects, seed=seed
        )
    objects, targets = scalar_set
    result = scalar_head_metrics(
        model, objects, targets, target=target, seq_len=seq_len, objects_per_batch=objects_per_batch
    )
    result["checkpoint"] = str(checkpoint)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--target", default="Z")
    parser.add_argument("--n-objects", type=int, default=2048)
    parser.add_argument("--seq-len", type=int, default=896)
    parser.add_argument("--objects-per-batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = evaluate_checkpoint(
        args.checkpoint,
        args.data_root,
        target=args.target,
        n_objects=args.n_objects,
        seq_len=args.seq_len,
        objects_per_batch=args.objects_per_batch,
        device=args.device,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
