"""Phase 4 GPU verification (gpu-marked; excluded from the default CPU run).

Covers the plan's Phase 4 gates on tiny synthetic runs:

- Pythia schedule: checkpoint dirs at exactly steps 1,2,4,...,512,1000; each
  carries streaming-dataset state; each converts to HF and loads via
  ``AutoModel.from_pretrained`` (driven through ``run_probe_sweep.py``, which
  also produces val-loss + ridge-probe entries per checkpoint);
- kill/resume: a run checkpointed at step 137 and SIGKILLed later, then
  resumed, overlays the uninterrupted 200-step loss curve and consumes
  exactly the same object stream (object_id log: no replay, no gap).

Run (training machine / reserved GPU):
    pytest -m gpu tests/test_phase4_gpu.py -v
"""

import json
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.gpu

ASTRO_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ASTRO_DIR.parent
NANOTRON_DIR = REPO_ROOT / "nanotron"

PYTHIA_TINY_STEPS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000]
LOSS_RE = re.compile(r"lm_loss: ([0-9.eE+-]+)")


def load_tiny_config():
    return yaml.safe_load(
        (ASTRO_DIR / "configs" / "nanotron" / "astropt3-test-tiny.yaml").read_text()
    )


def run_train(config: dict, workdir: Path, name: str, timeout: int = 5400) -> str:
    config_path = workdir / f"{name}.yaml"
    config_path.write_text(yaml.safe_dump(config))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=1",
            "--rdzv-backend=c10d",
            "--rdzv-endpoint=localhost:0",
            str(NANOTRON_DIR / "run_train.py"),
            "--config-file",
            str(config_path),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "CUDA_DEVICE_MAX_CONNECTIONS": "1"},
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-4000:]}"
    return result.stdout


def losses_from(stdout: str) -> list[float]:
    return [float(m) for m in LOSS_RE.findall(stdout)]


def test_pythia_schedule_conversions_and_probe_sweep(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("pythia_run")
    ckpt_dir = workdir / "checkpoints"

    config = load_tiny_config()
    config["tokens"]["train_steps"] = 1000
    config["checkpoints"].update(
        {
            "checkpoints_path": str(ckpt_dir),
            "checkpoint_interval": 1000,
            "checkpoint_schedule": "pythia",
            "save_final_state": False,
        }
    )
    config["optimizer"]["learning_rate_scheduler"]["lr_decay_steps"] = 990
    config["logging"]["iteration_step_info_interval"] = 50
    run_train(config, workdir, "pythia-tiny")

    # checkpoint dirs at exactly the Pythia steps, each with dataset state
    step_dirs = sorted(int(p.name) for p in ckpt_dir.iterdir() if p.is_dir() and p.name.isdigit())
    assert step_dirs == PYTHIA_TINY_STEPS
    for step in step_dirs:
        assert (ckpt_dir / str(step) / "dataset_state" / "dp_0.pt").exists(), step

    # async sweep: convert every checkpoint, val-loss + probe each
    sweep_dir = workdir / "sweep"
    result = subprocess.run(
        [
            sys.executable,
            str(ASTRO_DIR / "scripts" / "run_probe_sweep.py"),
            f"--checkpoints-dir={ckpt_dir}",
            f"--out-dir={sweep_dir}",
            "--data-root=synthetic",
            "--val-batches=32",
            "--probe-objects=256",
            "--micro-batch-size=4",
            "--seq-len=896",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=5400,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-4000:]}"

    rows = [
        json.loads(line)
        for line in (sweep_dir / "probe_results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    by_step = {row["step"]: row for row in rows}
    assert sorted(by_step) == PYTHIA_TINY_STEPS
    for row in rows:
        assert math.isfinite(row["val_loss"]), row
        assert math.isfinite(row["probe_r2"]) and row["probe_r2"] <= 1.0, row
    # training must show up in the fixed-batch val loss
    assert by_step[1000]["val_loss"] < by_step[1]["val_loss"], (
        by_step[1]["val_loss"],
        by_step[1000]["val_loss"],
    )

    # every converted checkpoint loads through the Auto classes
    import astropt3  # noqa: F401

    from transformers import AutoModel

    for step in PYTHIA_TINY_STEPS:
        model = AutoModel.from_pretrained(sweep_dir / "hf" / str(step))
        assert model.config.model_type == "astropt3", step

    print("pythia sweep:", json.dumps({s: (by_step[s]["val_loss"], by_step[s]["probe_r2"]) for s in by_step}))


def kill_resume_config(base_dir: Path, log_name: str, train_steps: int, interval: int) -> dict:
    config = load_tiny_config()
    config["tokens"]["train_steps"] = train_steps
    config["checkpoints"].update(
        {
            "checkpoints_path": str(base_dir / "checkpoints"),
            "checkpoint_interval": interval,
            "save_final_state": False,
        }
    )
    config["optimizer"]["learning_rate_scheduler"]["lr_decay_steps"] = 190
    config["data_stages"][0]["data"]["dataset"]["object_id_log"] = str(base_dir / log_name)
    return config


def test_kill_at_137_resume_overlays_loss_curve(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("kill_resume")

    # reference: uninterrupted 200 steps, no intermediate checkpoints
    ref_dir = workdir / "ref"
    ref_dir.mkdir()
    config_a = kill_resume_config(ref_dir, "objects_a.log", train_steps=200, interval=1000)
    stdout_a = run_train(config_a, ref_dir, "ref-200")
    losses_a = losses_from(stdout_a)
    assert len(losses_a) == 200

    # interrupted: checkpoint at 137, SIGKILL once ~15 further steps ran
    kill_dir = workdir / "killed"
    kill_dir.mkdir()
    config_b = kill_resume_config(kill_dir, "objects_b1.log", train_steps=200, interval=137)
    config_path = kill_dir / "killed.yaml"
    config_path.write_text(yaml.safe_dump(config_b))
    out_file = open(kill_dir / "stdout.log", "w")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=1",
            "--rdzv-backend=c10d",
            "--rdzv-endpoint=localhost:0",
            str(NANOTRON_DIR / "run_train.py"),
            "--config-file",
            str(config_path),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "CUDA_DEVICE_MAX_CONNECTIONS": "1"},
        stdout=out_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    latest = kill_dir / "checkpoints" / "latest.txt"
    deadline = time.time() + 3600
    try:
        while time.time() < deadline:
            done_steps = len(losses_from((kill_dir / "stdout.log").read_text()))
            if latest.exists() and latest.read_text().strip() == "137" and done_steps >= 140:
                break  # checkpoint written and a few steps past it -> kill now
            if proc.poll() is not None:
                # tiny steps are fast: the run may finish before the poll sees
                # step 140 — acceptable as long as the 137 checkpoint exists
                assert proc.returncode == 0 and latest.exists() and latest.read_text().strip() == "137", (
                    f"run died before checkpoint 137: {(kill_dir / 'stdout.log').read_text()[-4000:]}"
                )
                break
            time.sleep(0.5)
        else:
            pytest.fail("timed out waiting for checkpoint 137")
    finally:
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=60)
        out_file.close()
    losses_b1 = losses_from((kill_dir / "stdout.log").read_text())
    assert len(losses_b1) >= 138, f"killed run logged only {len(losses_b1)} steps"

    # resume from the 137 checkpoint and finish the 200 steps
    config_c = kill_resume_config(kill_dir, "objects_b2.log", train_steps=200, interval=1000)
    config_c["checkpoints"]["resume_checkpoint_path"] = str(kill_dir / "checkpoints")
    stdout_c = run_train(config_c, kill_dir, "resumed")
    losses_c = losses_from(stdout_c)
    assert len(losses_c) == 63, f"expected steps 138..200, got {len(losses_c)} loss lines"

    # exact-resume check: the resumed run must continue ITS OWN run's
    # trajectory — same restored weights, same data. The step-138 loss is a
    # pure forward on the restored state (computed before the backward), so
    # it must match at log precision; later steps drift slowly because the
    # flash-attn backward is nondeterministic even within a single run
    overlap = min(len(losses_b1) - 137, len(losses_c), 10)
    assert overlap >= 1
    assert losses_c[0] == pytest.approx(losses_b1[137], rel=2e-3), (losses_c[0], losses_b1[137])
    for i in range(1, overlap):
        assert losses_c[i] == pytest.approx(losses_b1[137 + i], rel=5e-2), (
            i,
            losses_c[i],
            losses_b1[137 + i],
        )

    # coarse overlay vs the INDEPENDENT uninterrupted run: identical data
    # stream, but run-to-run kernel nondeterminism compounds over 137 prior
    # steps, so only a loose band is meaningful here
    tail_a = losses_a[137:]
    rel = [abs(c - a) / a for c, a in zip(losses_c, tail_a)]
    assert sum(rel) / len(rel) < 0.10, f"mean relative loss deviation {sum(rel) / len(rel):.4f}"
    print(
        f"resume overlay: own-run overlap {overlap} steps "
        f"(first {losses_c[0]} vs {losses_b1[137]}), "
        f"independent-run mean rel dev {sum(rel) / len(rel):.4f}"
    )

    # object stream: resumed log is exactly the uninterrupted log's tail —
    # the stream continued without replaying any trained object
    lines_a = (ref_dir / "objects_a.log.dp0").read_text().splitlines()
    lines_b2 = (kill_dir / "objects_b2.log.dp0").read_text().splitlines()
    assert 0 < len(lines_b2) < len(lines_a)
    assert lines_a[-len(lines_b2) :] == lines_b2, "resumed stream != uninterrupted continuation"
    assert not set(lines_a[: len(lines_a) - len(lines_b2)]) & set(lines_b2), "resume replayed objects"
    # and the killed run consumed the same prefix while it lived
    lines_b1 = (kill_dir / "objects_b1.log.dp0").read_text().splitlines()
    assert lines_b1[: len(lines_a) - len(lines_b2)] == lines_a[: len(lines_a) - len(lines_b2)]
