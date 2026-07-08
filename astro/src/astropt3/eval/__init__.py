"""Offline evaluation of converted HF checkpoints (Phase 4).

``val_loss`` recomputes the training objective on a fixed, deterministic set
of validation micro-batches; ``linear_probe`` ridge-regresses redshift from
mean-pooled hidden states. Both run on a converted HF checkpoint, never
inside the nanotron trainer — evaluation must not block training
(``scripts/run_probe_sweep.py`` drives them asynchronously as checkpoints
land).
"""
