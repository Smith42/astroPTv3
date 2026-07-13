"""JetFormer-tokeniser GPU verification (gpu-marked; excluded from CPU runs).

Covers the jetformer plan's J3 gates (astro/docs/jetformer_plan.md):

1. HF <-> nanotron loss parity on a fixed synthetic batch after conversion,
   eval mode (noise off), both conversion directions;
2. TP=2: flow + GMM-head gradients bit-identical across ranks, with the
   noise curriculum ACTIVE (drawn under the tp_synced random state);
3. 50-step CUDA smoke — the likelihood loss (NLL - logdet, may be negative)
   decreases; the final checkpoint converts to HF and reproduces the loss;
4. kill/resume with a jetformer config reproduces the exact object stream
   (the loss path is orthogonal to stream state — cheap regression guard;
   no per-step loss overlay: RNG states are not restored on resume, so the
   resumed run draws fresh curriculum noise, and the tiny-run likelihood
   loss is tens of nats noisy around its zero crossing).

Run (reserved GPU node, [train] env):
    pytest -m gpu tests/test_jetformer_gpu.py -v
"""

import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
import yaml

from test_nanotron_gpu import (
    REL_TOL,
    build_nt_model,
    make_nt,
    micro_batch,
    regroup,
    tiny_nt_config,
)
from test_phase4_gpu import losses_from, run_train

pytestmark = pytest.mark.gpu

ASTRO_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ASTRO_DIR.parent
NANOTRON_DIR = REPO_ROOT / "nanotron"
TOOLS_DIR = NANOTRON_DIR / "tools" / "astropt3"
JET_YAML = ASTRO_DIR / "configs" / "nanotron" / "astropt3-test-tiny-jetformer.yaml"


@pytest.fixture(scope="session")
def nt():
    return make_nt()


def tiny_jet_config(nt):
    return tiny_nt_config(nt, tokeniser="jetformer", jetformer_flow_hidden=32)


@pytest.fixture(scope="session")
def matched_jet_models(nt):
    """HF tiny jetformer model and a nanotron model carrying identical weights."""
    import astropt3  # noqa: F401 -- registers Auto classes

    from astropt3 import AstroPT3Model

    nt_config = tiny_jet_config(nt)
    hf_config = nt.to_hf.get_hf_config(nt_config)
    assert hf_config.tokeniser == "jetformer"
    assert hf_config.jetformer_flow_hidden == 32
    torch.manual_seed(0)
    hf_model = AstroPT3Model(hf_config)
    nt_model = build_nt_model(nt, nt_config)
    nt.from_hf.convert_hf_to_nt(hf_model, nt_model, nt_config)
    hf_model = hf_model.cuda().to(torch.bfloat16).eval()
    nt_model.eval()  # noise curriculum off: deterministic parity
    return hf_model, nt_model, nt_config, hf_config


def test_forward_loss_parity(nt, matched_jet_models):
    hf_model, nt_model, nt_config, hf_config = matched_jet_models
    flat = micro_batch(hf_config)
    names = hf_config.modality_registry().names()
    with torch.no_grad():
        nt_out = nt_model(**flat)
        hf_out = hf_model(**regroup(flat, names))

    nt_loss = nt_out["loss"].item()
    hf_loss = hf_out.loss.item()
    assert nt_loss == pytest.approx(hf_loss, rel=REL_TOL), (nt_loss, hf_loss)
    for name in names:
        nt_mod = nt_out[f"{name}_loss"].item()
        hf_mod = hf_out.modality_losses[name].item()
        assert nt_mod == pytest.approx(hf_mod, rel=REL_TOL), (name, nt_mod, hf_mod)


def test_conversion_roundtrip_exact(nt, matched_jet_models):
    from astropt3 import AstroPT3Model

    hf_model, nt_model, nt_config, hf_config = matched_jet_models
    hf_back = AstroPT3Model(hf_config).cuda().to(torch.bfloat16)
    nt.to_hf.convert_nt_to_hf(nt_model, hf_back, nt_config)
    original = hf_model.state_dict()
    back = hf_back.state_dict()
    assert set(back) == set(original)
    for key, value in back.items():
        assert torch.equal(value, original[key]), f"roundtrip drift in {key}"
    # the flow weights actually took part (not just body params)
    assert any(key.startswith("flows.") for key in back), "no flow params converted"
    assert any("decoders." in key and ".proj." in key for key in back), "no GMM heads converted"


def test_tp2_replicated_module_gradients_with_noise(nt):
    if torch.cuda.device_count() < 2:
        pytest.skip("TP=2 check needs 2 visible GPUs")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=2",
            "--rdzv-backend=c10d",
            "--rdzv-endpoint=localhost:0",
            str(ASTRO_DIR / "scripts" / "tp2_grad_check.py"),
            "--tokeniser=jetformer",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "CUDA_DEVICE_MAX_CONNECTIONS": "1"},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "TP2 GRAD CHECK PASS (jetformer" in result.stdout


def test_50step_synthetic_run_and_checkpoint_conversion(nt, tmp_path_factory):
    """torchrun 50 jetformer steps on synthetic data; convert; reproduce loss."""
    workdir = tmp_path_factory.mktemp("jetformer_smoke")
    config = yaml.safe_load(JET_YAML.read_text())
    config["checkpoints"]["checkpoints_path"] = str(workdir / "checkpoints")
    stdout = run_train(config, workdir, "jet-tiny")

    losses = losses_from(stdout)
    assert len(losses) >= 40, f"expected ~50 logged losses, got {len(losses)}"
    # likelihood loss can cross zero: assert an absolute drop, not a ratio
    assert losses[-1] < losses[0] - 10.0, f"jetformer smoke did not learn: {losses[0]:.4f} -> {losses[-1]:.4f}"

    checkpoint = workdir / "checkpoints" / "50"
    assert (checkpoint / "model_config.json").exists()

    save_path = workdir / "hf"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=1",
            "--rdzv-backend=c10d",
            "--rdzv-endpoint=localhost:0",
            str(TOOLS_DIR / "convert_nanotron_to_hf.py"),
            f"--checkpoint_path={checkpoint}",
            f"--save_path={save_path}",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "CUDA_DEVICE_MAX_CONNECTIONS": "1"},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    import astropt3  # noqa: F401
    import json

    from transformers import AutoModel

    hf_model = AutoModel.from_pretrained(save_path).cuda().to(torch.bfloat16).eval()
    assert hf_model.config.tokeniser == "jetformer"

    with open(checkpoint / "model_config.json") as f:
        nt_config = nt.config_cls(**json.load(f))
    nt_model = build_nt_model(nt, nt_config)
    nt.serialize.load_weights(
        model=nt_model, parallel_context=nt.parallel_context, root_folder=checkpoint
    )
    nt_model.eval()
    flat = micro_batch(hf_model.config, skip=7)  # not the first training batch
    names = hf_model.config.modality_registry().names()
    with torch.no_grad():
        nt_loss = nt_model(**flat)["loss"].item()
        hf_loss = hf_model(**regroup(flat, names)).loss.item()
    assert hf_loss == pytest.approx(nt_loss, rel=REL_TOL), (nt_loss, hf_loss)


def _jet_run_config(base_dir: Path, log_name: str, train_steps: int, interval: int) -> dict:
    config = yaml.safe_load(JET_YAML.read_text())
    config["tokens"]["train_steps"] = train_steps
    config["checkpoints"].update(
        {
            "checkpoints_path": str(base_dir / "checkpoints"),
            "checkpoint_interval": interval,
            "save_final_state": False,
        }
    )
    config["optimizer"]["learning_rate_scheduler"]["lr_decay_steps"] = train_steps - 10
    config["data_stages"][0]["data"]["dataset"]["object_id_log"] = str(base_dir / log_name)
    return config


def test_kill_resume_reproduces_object_stream(tmp_path_factory):
    """Kill after checkpoint 60 (the only interval multiple < 100), resume to 100."""
    workdir = tmp_path_factory.mktemp("jet_kill_resume")

    # reference: uninterrupted run, for the stream-continuation ledger
    ref_dir = workdir / "ref"
    ref_dir.mkdir()
    run_train(_jet_run_config(ref_dir, "objects_a.log", 100, interval=1000), ref_dir, "ref-100")

    # interrupted: checkpoint at 60 (the 2nd multiple, 120, exceeds the step
    # budget so latest.txt cannot advance past 60 before the kill), SIGKILL later
    kill_dir = workdir / "killed"
    kill_dir.mkdir()
    config_b = _jet_run_config(kill_dir, "objects_b1.log", 100, interval=60)
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
            if latest.exists() and latest.read_text().strip() == "60" and done_steps >= 68:
                break
            if proc.poll() is not None:
                assert proc.returncode == 0 and latest.exists() and latest.read_text().strip() == "60", (
                    f"run died before checkpoint 60: {(kill_dir / 'stdout.log').read_text()[-4000:]}"
                )
                break
            time.sleep(0.5)
        else:
            pytest.fail("timed out waiting for checkpoint 60")
    finally:
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=60)
        out_file.close()

    assert (kill_dir / "checkpoints" / "60" / "dataset_state" / "dp_0.pt").exists()

    # resume from 60 and finish
    config_c = _jet_run_config(kill_dir, "objects_b2.log", 100, interval=1000)
    config_c["checkpoints"]["resume_checkpoint_path"] = str(kill_dir / "checkpoints")
    stdout_c = run_train(config_c, kill_dir, "resumed")
    losses_c = losses_from(stdout_c)
    assert len(losses_c) == 40, f"expected steps 61..100, got {len(losses_c)} loss lines"

    # exact object-stream continuation: the resumed log is exactly the
    # uninterrupted log's tail, with no replay and no gap
    lines_a = (ref_dir / "objects_a.log.dp0").read_text().splitlines()
    lines_b2 = (kill_dir / "objects_b2.log.dp0").read_text().splitlines()
    assert 0 < len(lines_b2) < len(lines_a)
    assert lines_a[-len(lines_b2) :] == lines_b2, "resumed stream != uninterrupted continuation"
    assert not set(lines_a[: len(lines_a) - len(lines_b2)]) & set(lines_b2), "resume replayed objects"

    # no per-step loss overlay: RNG states are not restored on resume (fresh
    # curriculum noise) and the tiny-run likelihood loss swings tens of nats
    # step-to-step around its zero crossing, so per-step comparison carries no
    # signal. The stream check above is the exact gate; here only assert the
    # resumed trajectory is finite and coarsely trained (well below the
    # early-run loss scale).
    losses_b1 = losses_from((kill_dir / "stdout.log").read_text())
    assert all(math.isfinite(loss) for loss in losses_c)
    early = sum(losses_b1[:10]) / 10
    late = sum(losses_c) / len(losses_c)
    assert late < early - 10.0, f"resumed losses not trained: early {early:.1f} vs resumed {late:.1f}"
    print(f"jetformer kill/resume: stream exact; resumed mean loss {late:.1f} (early {early:.1f})")
