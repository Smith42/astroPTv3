"""Shard writer/reader roundtrip, sharding disjointness, and split assignment."""

import numpy as np
import pytest
import torch

from astropt3.data import mmu
from astropt3.data.synthetic import IMAGE_BANDS, make_record, record_stream

N_RECORDS = 64
SHARD_SIZE = 8  # -> 8 shards, divisible by 2 ranks x 2 workers


@pytest.fixture(scope="module")
def shard_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("pilot_shards")
    records = list(record_stream(N_RECORDS))
    for k in range(0, N_RECORDS, SHARD_SIZE):
        mmu.write_shard(records[k : k + SHARD_SIZE], out / f"shard-{k:05d}.parquet")
    assert len(sorted(out.glob("shard-*.parquet"))) == N_RECORDS // SHARD_SIZE
    return out


def test_roundtrip_matches_source_records(shard_dir):
    decoded = list(mmu.MMUIterableDataset(shard_dir))
    assert len(decoded) == N_RECORDS
    for i, rec in enumerate(decoded):
        source = make_record(i)
        assert rec["object_id"] == source["object_id"]
        flux = rec["image"]["flux"]
        assert flux.shape == mmu.IMAGE_SHAPE and flux.dtype == np.float32
        np.testing.assert_array_equal(flux, source["image"]["flux"])
        assert ("spectrum" in rec) == ("spectrum" in source)
        if "spectrum" in rec:
            np.testing.assert_array_equal(
                rec["spectrum"]["flux"], source["spectrum"]["flux"]
            )
            assert rec["spectrum"]["mask"].dtype == bool
            assert rec["spectrum"]["flux"].shape == (mmu.SPECTRUM_LENGTH,)
            assert rec["Z"] == pytest.approx(source["Z"])


def test_spectrum_only_roundtrip(tmp_path):
    # ADR 0005: a spectrum-only record survives write/decode with no image key
    records = [
        make_record(3),
        make_record(3, image_only_fraction=0.0, spectrum_only_fraction=1.0),
    ]
    mmu.write_shard(records, tmp_path / "shard-00000.parquet")
    decoded = list(mmu.MMUIterableDataset(tmp_path))
    assert "image" in decoded[0] and "image" not in decoded[1]
    assert "spectrum" in decoded[1] and "z_spec" not in records[1]
    np.testing.assert_array_equal(
        decoded[1]["spectrum"]["flux"], records[1]["spectrum"]["flux"]
    )


def test_spectra_subdir_oversampled(shard_dir, tmp_path):
    # spectra/ shards are listed spectra_repeat times per epoch (ADR 0005)
    spec_records = [
        make_record(i, image_only_fraction=0.0, spectrum_only_fraction=1.0)
        for i in range(100, 104)
    ]
    for f in sorted(shard_dir.glob("*.parquet")):
        (tmp_path / f.name).symlink_to(f)
    mmu.write_shard(spec_records, tmp_path / "spectra" / "shard-00000.parquet")

    dataset = mmu.MMUIterableDataset(tmp_path, spectra_repeat=3)
    assert dataset.n_shards == N_RECORDS // SHARD_SIZE + 3
    ids = [r["object_id"] for r in dataset]
    assert len(ids) == N_RECORDS + 3 * len(spec_records)
    for rec in spec_records:
        assert ids.count(rec["object_id"]) == 3

    # repeat=1 includes the spectrum-only shard exactly once
    ids1 = [r["object_id"] for r in mmu.MMUIterableDataset(tmp_path)]
    assert len(ids1) == N_RECORDS + len(spec_records)


def test_normalize_record_coerces_synthetic_schema():
    normalized = mmu.normalize_record(make_record(3))
    image = normalized["image"]
    assert image["band"] == IMAGE_BANDS
    assert len(image["psf_fwhm"]) == mmu.N_BANDS  # scalar -> per-band
    assert normalized["match_dist_arcsec"] is None
    assert normalized["ZWARN"] is False  # synthetic spectrum rows carry the good flag


def test_rank_worker_sharding_disjoint_and_complete(shard_dir):
    all_ids = {make_record(i)["object_id"] for i in range(N_RECORDS)}
    per_rank = []
    for rank in range(2):
        dataset = mmu.MMUIterableDataset(shard_dir, rank=rank, world_size=2)
        assert dataset.n_shards == 4
        loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=2)
        per_rank.append([rec["object_id"] for rec in loader])
    assert set(per_rank[0]).isdisjoint(per_rank[1])
    assert set(per_rank[0]) | set(per_rank[1]) == all_ids
    assert len(per_rank[0]) + len(per_rank[1]) == N_RECORDS


def test_decoded_records_feed_sequencer(shard_dir, sequencer, collator):
    objects = []
    for i, record in enumerate(mmu.MMUIterableDataset(shard_dir)):
        if i >= 4:
            break
        objects.append(sequencer.build(record))
    batch = collator(objects)
    assert batch["input_ids"].shape[1] == 896
    for values in batch["modality_values"].values():
        assert torch.isfinite(values).all()


def test_shuffle_reorders_but_preserves_contents_and_shards(shard_dir):
    plain = [r["object_id"] for r in mmu.MMUIterableDataset(shard_dir)]
    dataset = mmu.MMUIterableDataset(shard_dir, shuffle_buffer_size=32, seed=1)
    # datasets 5.x .shuffle() would collapse this to 1, breaking worker splits
    assert dataset.n_shards == N_RECORDS // SHARD_SIZE
    shuffled = [r["object_id"] for r in dataset]
    assert shuffled != plain
    assert sorted(shuffled) == sorted(plain)

    assert [r["object_id"] for r in dataset] == shuffled  # deterministic
    dataset.set_epoch(1)
    epoch1 = [r["object_id"] for r in dataset]
    assert epoch1 != shuffled and sorted(epoch1) == sorted(plain)


def test_tmp_staging_files_excluded_from_stream(tmp_path):
    mmu.write_shard(list(record_stream(4)), tmp_path / "shard-00000.parquet")
    # a crashed write leaves only a .tmp file behind; the reader must skip it
    (tmp_path / "shard-00001.parquet.tmp").write_bytes(b"partial garbage")
    ids = [r["object_id"] for r in mmu.MMUIterableDataset(tmp_path)]
    assert len(ids) == 4


def test_assign_split_deterministic_and_spatially_coherent():
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 2**59, size=20_000)
    splits = [mmu.assign_split(p) for p in pixels]
    assert splits == [mmu.assign_split(p) for p in pixels]
    val_fraction = splits.count("val") / len(splits)
    assert 0.01 < val_fraction < 0.03  # ~2% of order-7 tiles

    # all descendants of one order-7 tile land in the same split
    base = int(pixels[0]) >> (2 * (29 - 7)) << (2 * (29 - 7))
    offsets = rng.integers(0, 4 ** (29 - 7), size=50)
    assert len({mmu.assign_split(base + int(o)) for o in offsets}) == 1

    # the salt actually reshuffles the assignment
    assert any(
        mmu.assign_split(p) != mmu.assign_split(p, salt=1) for p in pixels[:1000]
    )
