"""Source-distinct sequence contracts for all ADR 0013 spokes."""

from pathlib import Path

import torch
import yaml

from astropt3 import AstroPT3Config, AstroPT3Model
from astropt3.config_io import load_model_config
from astropt3.data.packing import ObjectSequencer, PackedCollator
from astropt3.data.scalar_registry import GWH_FRACTION_FIELDS
from astropt3.data.synthetic import make_record

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "model"


def test_sdss_and_hsc_use_distinct_source_modalities(tiny_config):
    modalities = tiny_config.to_dict()["modalities"] + [
        {
            "name": "sdss_spectra",
            "family": "spectrum",
            "source": "sdss",
            "record_keys": ["sdss_spectrum"],
            "input_size": 256,
            "patch_size": 256,
            "pos_type": "continuous",
            "pos_input_size": 1,
        },
        {
            "name": "hsc_images",
            "family": "image",
            "source": "hsc",
            "record_keys": ["hsc_image"],
            "input_size": 320,
            "patch_size": 8,
            "pos_type": "index",
            "max_positions": 144,
        },
    ]
    config = AstroPT3Config(
        **{
            **tiny_config.to_dict(),
            "modalities": modalities,
            "tokeniser": "jetformer",
        }
    )
    record = make_record(9, image_only_fraction=1.0)
    sdss_lambda = 10 ** (torch.log10(torch.tensor(3800.0)) + 1e-4 * torch.arange(3800))
    record["sdss_spectrum"] = {
        "flux": torch.ones(3800),
        "lambda": sdss_lambda,
        "mask": torch.zeros(3800, dtype=torch.bool),
    }
    record["hsc_image"] = {
        "flux": torch.ones(5, 160, 160),
        "band": ["hsc-g", "hsc-r", "hsc-i", "hsc-z", "hsc-y"],
    }

    sequence = ObjectSequencer(config).build(record, include_scalars=False)
    assert list(sequence.values["sdss_spectra"].shape) == [15, 256]
    assert list(sequence.values["hsc_images"].shape) == [144, 320]
    assert config.modality_registry().get_config("sdss_spectra").token_ids != (
        config.modality_registry().get_config("hsc_images").token_ids
    )


def test_complete_source_graph_config_sequences_every_accepted_target(tmp_path):
    config, _ = load_model_config(CONFIGS / "test-tiny-source-graph.yaml")
    record = make_record(9, image_only_fraction=1.0)
    sdss_lambda = 10 ** (torch.log10(torch.tensor(3800.0)) + 1e-4 * torch.arange(3800))
    record["sdss_spectrum"] = {
        "flux": torch.ones(3800),
        "lambda": sdss_lambda,
        "mask": torch.zeros(3800, dtype=torch.bool),
    }
    record["hsc_image"] = {
        "flux": torch.ones(5, 160, 160),
        "band": ["hsc-g", "hsc-r", "hsc-i", "hsc-z", "hsc-y"],
    }
    record["sdss_Z"] = 0.2
    record.update(
        {
            "provabgs_LOG_MSTAR": 10.0,
            "provabgs_Z_HP": 0.2,
            "provabgs_Z_MW": 0.01,
            "provabgs_TAGE_MW": 8.0,
            "provabgs_AVG_SFR": 1.0,
        }
    )
    record.update({f"gwh_{field}": 0.5 for field in GWH_FRACTION_FIELDS})
    # ADR 0014 A8 free scalars: the anchor's arrive from make_record, the HSC
    # ones ride on the partner row we already fetched
    record.update({f"hsc_{band}_cmodel_mag": 21.0 for band in "grizy"})
    record["hsc_extendedness"] = 1.0
    record.update({f"hsc_i_sdssshape_shape{m}": 0.3 for m in ("11", "22", "12")})
    record.update({f"hsc_i_sdssshape_psf_shape{m}": 0.11 for m in ("11", "22", "12")})

    sequence = ObjectSequencer(config).build(record)
    expected = {
        "images",
        "sdss_spectra",
        "hsc_images",
        "sdss_Z",
        "provabgs_LOG_MSTAR",
        *(f"gwh_{field}" for field in GWH_FRACTION_FIELDS),
        # A8
        "fiberflux",
        "psfdepth",
        "psf_fwhm",
        "hsc_cmodel_mag",
        "hsc_extendedness",
        "hsc_shape",
        "hsc_psf_shape",
    }
    assert expected <= sequence.masks.keys()
    assert len(config.modalities) == 54  # 47 + A8's seven
    assert (
        len({token for mod in config.modalities for token in mod["token_ids"]}) == 162
    )
    assert len(sequence) < 512

    model = AstroPT3Model(config)
    batch = PackedCollator(config, seq_len=512)([sequence])
    with torch.no_grad():
        output = model(**batch)
    assert torch.isfinite(output.loss)
    assert {"image", "spectrum", "scalar"} == output.family_losses.keys()
    assert output.modality_losses.keys() == sequence.masks.keys()

    from astropt3.eval.samples import render_sampled_tokens

    rendered = render_sampled_tokens(
        model,
        record,
        sequence,
        {
            "sdss_spectra": sequence.values["sdss_spectra"].unsqueeze(0),
            "hsc_images": sequence.values["hsc_images"].unsqueeze(0),
        },
        out_dir=tmp_path,
        tag="sources",
        show_truth=True,
    )
    assert rendered.keys() == {"sdss_spectra", "hsc_images"}
    assert all(path.exists() for path in rendered.values())


def test_nanotron_source_graph_config_has_the_same_token_map():
    config, _ = load_model_config(CONFIGS / "test-tiny-source-graph.yaml")
    path = (
        CONFIGS.parent / "nanotron" / "astropt3-70m-jetformer-north-5spoke-replay2.yaml"
    )
    nanotron = yaml.safe_load(path.read_text())["model"]["model_config"]
    assert [mod["name"] for mod in nanotron["modalities"]] == [
        mod["name"] for mod in config.modalities
    ]
    assert [mod["token_ids"] for mod in nanotron["modalities"]] == [
        mod["token_ids"] for mod in config.modalities
    ]
    assert nanotron["vocab_size"] == config.vocab_size == 166  # +21 for A8
    assert nanotron["loss_aggregation"] == "family"
