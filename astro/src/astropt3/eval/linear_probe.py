"""Ridge linear probe: mean-pooled central hidden states -> redshift ``Z``.

Streams validation records (val shards, or held-out synthetic indices),
keeps those carrying the target scalar, embeds each object by mean-pooling
the model's CENTRAL layer state (layer ``num_hidden_layers // 2`` — the
astroPT convention; mid-depth decoder states probe better than the final
layer, whose job is next-token emission) over one modality's patch tokens,
and fits
a closed-form ridge regression (numpy, no sklearn). The regularizer is
chosen on an inner validation split; the reported R^2 is on a held-out test
split. Fully deterministic for a given checkpoint.

Usage:
    python -m astropt3.eval.linear_probe --checkpoint <hf_dir> \
        --data-root <val_shards_dir|synthetic> --target Z --n-objects 2048
"""

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import torch

from ..data.packing import ObjectSequencer, PackedCollator
from ..data.synthetic import make_record
from ..tokenization import BOS_ID, PAD_ID
from .val_loss import SYNTHETIC_VAL_OFFSET


def _val_records(data_root, seed=0):
    if data_root == "synthetic":
        i = SYNTHETIC_VAL_OFFSET
        while True:
            yield make_record(i)
            i += 1
    else:
        # ADR 0006: the reserved val partitions, streamed live. Deterministic
        # given the seed, so every checkpoint is probed on the same records.
        from ..data.streaming import open_stream

        yield from open_stream(split="val", seed=seed)


# record key that feeds each poolable modality (mirrors ObjectSequencer)
_POOL_RECORD_KEY = {"images": "image", "spectra": "spectrum"}


def collect_probe_objects(
    config, data_root, target, n_objects, *, seed=0, pool_modality="images", max_scan=None
):
    """First ``n_objects`` val objects that carry a finite ``target`` scalar.

    Objects lacking the ``pool_modality`` are skipped — spectrum-only DESI
    rows carry ``Z`` but have no image tokens to pool over. Both record
    sources are endless (ADR 0006 streams the val partitions in a loop), so
    the scan is bounded by ``max_scan`` records; if fewer than ``n_objects``
    qualify within it, all qualifying objects are used (with a warning). The
    stream is deterministic, so every checkpoint in a sweep probes the same
    set — collect once and pass it to :func:`probe_checkpoint` as
    ``probe_set``.
    """
    sequencer = ObjectSequencer(config)
    source_key = _POOL_RECORD_KEY.get(pool_modality)
    objects, targets = [], []
    budget = max_scan if max_scan is not None else 50 * n_objects
    for record, _ in zip(_val_records(data_root, seed=seed), range(budget)):
        value = record.get(target)
        if value is None or not math.isfinite(float(value)):
            continue
        if source_key is not None and record.get(source_key) is None:
            continue
        # scalar-free sequences (ADR 0008): a Z span in the pooled sequence
        # would turn probe R^2 into a copying metric and break comparability
        # with every pre-0008 run
        obj = sequencer.build(record, include_scalars=False)
        if pool_modality not in obj.masks:
            continue
        objects.append(obj)
        targets.append(float(value))
        if len(objects) >= n_objects:
            break
    if not objects:
        raise ValueError(
            f"no records carry target {target!r} with {pool_modality!r} tokens"
        )
    if len(objects) < n_objects:
        warnings.warn(
            f"val scan budget reached: probing {len(objects)}/{n_objects} records that "
            f"carry target {target!r} with {pool_modality!r} tokens",
            stacklevel=2,
        )
    return objects, np.asarray(targets, dtype=np.float64)


def load_or_collect_probe_objects(
    cache_path, config, data_root, target, n_objects, *, seed=0, pool_modality="images"
):
    """Disk-cached :func:`collect_probe_objects` (atomic tmp+rename write).

    The collection scan can read the whole val split, so sweeps persist its
    result under their out dir and every restart reloads it in seconds. The
    cache is keyed on the collection arguments; a mismatch re-collects and
    overwrites (delete the file to force a refresh after regenerating data).
    """
    key = {
        "data_root": str(data_root),
        "target": target,
        "n_objects": n_objects,
        "seed": seed,
        "pool_modality": pool_modality,
    }
    cache_path = Path(cache_path)
    if cache_path.exists():
        payload = torch.load(cache_path, weights_only=False)
        if payload.get("key") == key:
            return payload["objects"], payload["targets"]
        warnings.warn(
            f"probe cache {cache_path} was built with {payload.get('key')}, "
            f"not {key}; re-collecting",
            stacklevel=2,
        )
    objects, targets = collect_probe_objects(
        config, data_root, target, n_objects, seed=seed, pool_modality=pool_modality
    )
    tmp = cache_path.with_name(cache_path.name + ".tmp")
    torch.save({"key": key, "objects": objects, "targets": targets}, tmp)
    tmp.rename(cache_path)
    return objects, targets


@torch.no_grad()
def embed_objects(model, config, objects, *, seq_len=896, objects_per_batch=8, pool_modality="images"):
    """Mean-pool the CENTRAL layer state over one modality's tokens, per object.

    Central = ``hidden_states[num_hidden_layers // 2]`` (astroPT convention;
    embeddings sit at index 0, so this is the output of the middle block).
    Objects are packed with the shared collator; each object's span in a row
    starts at its ``<|bos|>`` and the packed row-major object order equals
    the input order, so embeddings align with the targets by construction.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    collator = PackedCollator(config, seq_len=seq_len)
    central = config.num_hidden_layers // 2
    features = []
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
        out = model(**kwargs, compute_loss=False, output_hidden_states=True)
        hidden = out.hidden_states[central]  # [B, T, H]
        input_ids = batch["input_ids"]
        mask = batch["modality_masks"].get(pool_modality)
        if mask is None:
            raise ValueError(f"no {pool_modality!r} tokens in probe batch")
        for b in range(input_ids.shape[0]):
            starts = (input_ids[b] == BOS_ID).nonzero(as_tuple=True)[0].tolist()
            pad = (input_ids[b] == PAD_ID).nonzero(as_tuple=True)[0]
            end_of_row = int(pad[0]) if len(pad) else input_ids.shape[1]
            bounds = starts + [end_of_row]
            for s, e in zip(bounds[:-1], bounds[1:]):
                span_mask = mask[b, s:e]
                emb = hidden[b, s:e][span_mask].float().mean(dim=0)
                features.append(emb.cpu().numpy())
    features = np.asarray(features, dtype=np.float64)
    if len(features) != len(objects):
        raise RuntimeError(f"recovered {len(features)} embeddings for {len(objects)} objects")
    return features


def ridge_r2(X, y, *, seed=0, lambdas=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)):
    """Closed-form ridge with inner-split lambda selection; returns test R^2."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(X))
    n_test = max(1, len(X) // 5)
    n_val = max(1, (len(X) - n_test) // 5)
    test, val, train = order[:n_test], order[n_test : n_test + n_val], order[n_test + n_val :]

    mu, sigma = X[train].mean(axis=0), X[train].std(axis=0)
    sigma[sigma == 0] = 1.0
    Xn = (X - mu) / sigma
    y_mean = y[train].mean()

    def fit(idx, lam):
        A = Xn[idx].T @ Xn[idx] + lam * np.eye(Xn.shape[1])
        return np.linalg.solve(A, Xn[idx].T @ (y[idx] - y_mean))

    def r2(w, idx):
        pred = Xn[idx] @ w + y_mean
        ss_res = ((y[idx] - pred) ** 2).sum()
        ss_tot = ((y[idx] - y[idx].mean()) ** 2).sum()
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    best_lam = max(lambdas, key=lambda lam: r2(fit(train, lam), val))
    w = fit(np.concatenate([train, val]), best_lam)
    return {"r2": float(r2(w, test)), "lambda": float(best_lam), "n_test": int(n_test)}


def probe_checkpoint(
    checkpoint,
    data_root,
    *,
    target="Z",
    n_objects=2048,
    seq_len=896,
    objects_per_batch=8,
    pool_modality="images",
    device=None,
    seed=0,
    probe_set=None,
):
    """Probe one checkpoint; ``probe_set`` is an optional pre-collected
    ``(objects, targets)`` pair from :func:`collect_probe_objects` — the
    probe set depends only on the data stream, not on checkpoint weights,
    so sweeps should collect once and reuse it for every step."""
    import astropt3  # noqa: F401  -- registers the Auto classes

    from transformers import AutoModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = AutoModel.from_pretrained(checkpoint).to(device=device, dtype=dtype).eval()

    if probe_set is None:
        probe_set = collect_probe_objects(
            model.config, data_root, target, n_objects, seed=seed, pool_modality=pool_modality
        )
    objects, targets = probe_set
    X = embed_objects(
        model,
        model.config,
        objects,
        seq_len=seq_len,
        objects_per_batch=objects_per_batch,
        pool_modality=pool_modality,
    )
    result = ridge_r2(X, targets, seed=seed)
    result.update(
        {
            "checkpoint": str(checkpoint),
            "target": target,
            "n_objects": len(objects),
            "pool_modality": pool_modality,
        }
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--target", default="Z")
    parser.add_argument("--n-objects", type=int, default=2048)
    parser.add_argument("--seq-len", type=int, default=896)
    parser.add_argument("--objects-per-batch", type=int, default=8)
    parser.add_argument("--pool-modality", default="images")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    result = probe_checkpoint(
        args.checkpoint,
        args.data_root,
        target=args.target,
        n_objects=args.n_objects,
        seq_len=args.seq_len,
        objects_per_batch=args.objects_per_batch,
        pool_modality=args.pool_modality,
        device=args.device,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
