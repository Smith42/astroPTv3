"""Tiny plain-torch training loop on synthetic data.

This is a VALIDATION harness, not the trainer: real pretraining runs on the
nanotron fork (see the plan). It exists so the model + packing + loss stack
can be exercised end-to-end on CPU with zero network access.

Usage:
    python -m astropt3.train_smoke --config configs/model/test-tiny.yaml \
        --steps 50 --assert-decrease
"""

import argparse
import math
import time

import torch

from .config_io import load_model_config
from .data.packing import ObjectSequencer, PackedCollator
from .data.synthetic import record_stream
from .modeling_astropt3 import AstroPT3Model


def configure_optimizer(model, lr, weight_decay=0.1, betas=(0.9, 0.95)):
    """AdamW with weight decay on >=2D params only (astroPT convention)."""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


def lr_at(step, max_steps, peak_lr, warmup=10):
    """Linear warmup then cosine decay to 0.1x peak."""
    if step < warmup:
        return peak_lr * (step + 1) / warmup
    frac = (step - warmup) / max(max_steps - warmup, 1)
    return 0.1 * peak_lr + 0.5 * (peak_lr - 0.1 * peak_lr) * (1 + math.cos(math.pi * frac))


def make_batches(config, n_objects, objects_per_batch, seq_len):
    sequencer = ObjectSequencer(config)
    collator = PackedCollator(config, seq_len=seq_len)
    objs = [sequencer.build(r) for r in record_stream(n_objects)]
    batches = []
    for i in range(0, len(objs), objects_per_batch):
        chunk = objs[i : i + objects_per_batch]
        if chunk:
            batches.append(collator(chunk))
    return batches


def run(config_path, steps=50, objects_per_batch=4, seq_len=896, lr=3e-4, device="cpu", log_every=10):
    config, meta = load_model_config(config_path)
    torch.manual_seed(0)
    model = AstroPT3Model(config).to(device)
    model.train()
    opt = configure_optimizer(model, lr)

    n_objects = steps * objects_per_batch
    batches = make_batches(config, n_objects, objects_per_batch, seq_len)

    losses = []
    t0 = time.time()
    for step, batch in enumerate(batches[:steps]):
        batch = {
            k: (
                {kk: vv.to(device) for kk, vv in v.items()}
                if isinstance(v, dict)
                else v.to(device)
            )
            for k, v in batch.items()
        }
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, lr)
        if config.tokeniser == "jetformer":
            model.set_jet_noise_frac(step / steps)
        out = model(**batch)
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(out.loss.item())
        if step % log_every == 0:
            per_mod = {k: f"{v.item():.4f}" for k, v in out.modality_losses.items()}
            print(f"step {step:>4}  loss {losses[-1]:.4f}  {per_mod}  ({time.time() - t0:.1f}s)")
    print(f"final loss {losses[-1]:.4f} (initial {losses[0]:.4f}) after {len(losses)} steps")
    return losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--objects-per-batch", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=896)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--assert-decrease", action="store_true")
    args = parser.parse_args()

    losses = run(
        args.config,
        steps=args.steps,
        objects_per_batch=args.objects_per_batch,
        seq_len=args.seq_len,
        lr=args.lr,
        device=args.device,
    )
    if args.assert_decrease:
        assert losses[-1] < 0.7 * losses[0], (
            f"smoke training did not learn: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )
        print("PASS: final loss < 0.7x initial")


if __name__ == "__main__":
    main()
