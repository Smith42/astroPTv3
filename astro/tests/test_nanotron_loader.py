"""CPU tests for the nanotron micro-batch adapter (no nanotron import).

The adapter's output contract is what the nanotron fork's
``AstroPT3ForTraining.forward(**micro_batch)`` consumes; here the same flat
dicts are regrouped and fed to the HF model, which shares the packing/loss
semantics.
"""

from itertools import islice
from pathlib import Path

import mmu_stream.streaming as streaming
import pytest
import torch
from fake_mmu import fake_open_stream

from astropt3.data import nanotron_loader
from astropt3.data.nanotron_loader import (
    PackedMicroBatches,
)
from astropt3.data.nanotron_loader import (
    regroup_micro_batch as regroup,
)
from astropt3.tokenization import BOS_ID, modality_token_ids

MBS = 2
SEQ_LEN = 896


@pytest.fixture(scope="module")
def micro_batches(tiny_config):
    stream = PackedMicroBatches(tiny_config, MBS, SEQ_LEN)
    return list(islice(iter(stream), 3))


def test_micro_batch_contract(tiny_config, micro_batches):
    registry = tiny_config.modality_registry()
    for flat in micro_batches:
        assert flat["input_ids"].shape == (MBS, SEQ_LEN)
        assert flat["position_ids"].shape == (MBS, SEQ_LEN)
        assert flat["input_ids"].dtype == torch.long
        for name in registry.names():
            mod = registry.get_config(name)
            mask = flat[f"{name}_mask"]
            values = flat[f"{name}_values"]
            positions = flat[f"{name}_positions"]
            assert mask.shape == (MBS, SEQ_LEN) and mask.dtype == torch.bool
            assert values.shape == (int(mask.sum()), mod.input_size)
            assert values.dtype == torch.float32
            assert len(positions) == len(values)
            # placeholder ids sit exactly at the mask positions
            _, placeholder_id, _ = modality_token_ids(name)
            assert (flat["input_ids"][mask] == placeholder_id).all()
            # <|bos|> leads every object, so no modality token at position 0
            assert not mask[:, 0].any()
        # each row starts a fresh object: position_ids restart at 0
        assert (flat["position_ids"][:, 0] == 0).all()
        assert (flat["input_ids"][:, 0] == BOS_ID).all()


def test_batches_feed_hf_model(tiny_config, tiny_model, micro_batches):
    names = tiny_config.modality_registry().names()
    for flat in micro_batches:
        out = tiny_model(**regroup(flat, names))
        assert torch.isfinite(out.loss)


def test_absent_modality_ships_typed_empty_tensors(tiny_config, tiny_model):
    stream = PackedMicroBatches(
        tiny_config, MBS, SEQ_LEN, synthetic_image_only_fraction=1.0
    )
    flat = next(iter(stream))
    assert not flat["spectra_mask"].any()
    assert flat["spectra_values"].shape == (0, 256)
    assert flat["spectra_values"].dtype == torch.float32
    assert flat["spectra_positions"].shape == (0, 1)  # continuous positions
    assert flat["spectra_positions"].dtype == torch.float32
    # Z rides with the spectrum (ADR 0008): image-only records ship it empty
    assert flat["Z_values"].shape == (0, 1)
    assert flat["Z_positions"].dtype == torch.long
    out = tiny_model(**regroup(flat, tiny_config.modality_registry().names()))
    assert torch.isfinite(out.loss)
    assert set(out.modality_losses) == {"images", "ebv", "photometry"}


def test_synthetic_stream_disjoint_across_ranks_and_workers(tiny_config):
    # rank/worker sharding strides over record indices
    ds_a = PackedMicroBatches(tiny_config, MBS, SEQ_LEN, rank=0, world_size=2)
    ds_b = PackedMicroBatches(tiny_config, MBS, SEQ_LEN, rank=1, world_size=2)
    a = [r["object_id"] for r in islice(ds_a._synthetic_records(0, None), 20)]
    b = [r["object_id"] for r in islice(ds_b._synthetic_records(0, None), 20)]
    assert not set(a) & set(b)
    assert len(set(a)) == 20


def test_deterministic_across_instances(tiny_config):
    first = next(iter(PackedMicroBatches(tiny_config, MBS, SEQ_LEN)))
    second = next(iter(PackedMicroBatches(tiny_config, MBS, SEQ_LEN)))
    for key in first:
        assert torch.equal(first[key], second[key]), key


def test_mmu_stream_loops_epochs(tiny_config, monkeypatch):
    # the fake sources hold 24 records each: pulling many batches must cross
    # an epoch boundary without exhausting the endless stream
    monkeypatch.setattr("mmu_stream.streaming.open_stream", fake_open_stream)
    stream = PackedMicroBatches(
        tiny_config, MBS, SEQ_LEN, data_root="mmu", match_index="present"
    )
    batches = list(islice(iter(stream), 8))
    assert len(batches) == 8
    for flat in batches:
        assert flat["input_ids"].shape == (MBS, SEQ_LEN)


def test_more_workers_than_partitions_raises_a_named_error(tiny_config, monkeypatch):
    # datasets only WARNS and stops the surplus workers, so an over-subscribed
    # run trains on a fraction of its loaders. The fake corpus has 3 train
    # cells, so 4 workers must fail loudly with the remedy in the message.
    monkeypatch.setattr("mmu_stream.streaming.open_stream", fake_open_stream)
    stream = PackedMicroBatches(
        tiny_config, MBS, SEQ_LEN, data_root="mmu", match_index="present"
    )
    loader = torch.utils.data.DataLoader(
        stream, batch_size=None, num_workers=4, multiprocessing_context="fork"
    )
    with pytest.raises(ValueError, match="reduce num_loading_workers"):
        next(iter(loader))


def test_run_configs_fit_the_crossmatch_partition_ceiling():
    """dp x num_loading_workers must fit the ~165-cell crossmatch corpus."""
    import yaml

    # 173 cells in the published index minus streaming.VAL_PARTITIONS. Offline
    # constant on purpose; recompute with load_match_index() after a rebuild.
    ceiling = 173 - streaming.VAL_PARTITIONS
    configs = sorted(
        (Path(__file__).parents[1] / "configs" / "nanotron").glob("*.yaml")
    )
    assert configs, "no nanotron run configs found"
    for path in configs:
        config = yaml.safe_load(path.read_text())
        stage = config["data_stages"][0]["data"]
        if stage["dataset"].get("data_root") != "mmu":
            continue
        dp = config["parallelism"]["dp"]
        workers = stage["num_loading_workers"]
        # owned_by_rank deals the partitions, so the thinnest rank holds
        # floor(train / dp) — that is what caps this config's workers.
        assert workers <= ceiling // dp, (
            f"{path.name}: num_loading_workers({workers}) exceeds the "
            f"{ceiling // dp} partitions a dp({dp}) rank owns"
        )


class _FlakyStream:
    """Wraps a real fake stream to raise a 'client has been closed' RuntimeError
    once mid-iteration (the DNS-blip signature), delegating state_dict so the
    loader can resume the rebuilt stream from the pre-error snapshot."""

    def __init__(self, inner, fail_at):
        self._inner = inner
        self._fail_at = fail_at

    def __iter__(self):
        for i, rec in enumerate(self._inner):
            if self._fail_at is not None and i == self._fail_at:
                self._fail_at = None
                raise RuntimeError("client has been closed")
            yield rec

    def state_dict(self):
        return self._inner.state_dict()

    def load_state_dict(self, s):
        self._inner.load_state_dict(s)


def test_transient_error_rebuilds_and_reclaims(tiny_config, monkeypatch):
    # A DNS blip surfaces as this RuntimeError; the loader must ride it out by
    # rebuilding the stream AND reclaiming the abandoned one (gc.collect), or its
    # datasets/pyarrow prefetch buffers leak per rebuild to the cgroup OOM.
    builds = {"n": 0}

    def flaky(**kw):
        builds["n"] += 1
        return _FlakyStream(
            fake_open_stream(**kw), fail_at=5 if builds["n"] == 1 else None
        )

    collects = {"n": 0}
    real_collect = nanotron_loader.gc.collect

    def spy_collect(*a, **k):
        collects["n"] += 1
        return real_collect(*a, **k)

    monkeypatch.setattr("mmu_stream.streaming.open_stream", flaky)
    monkeypatch.setattr(nanotron_loader.time, "sleep", lambda *_: None)
    monkeypatch.setattr(nanotron_loader.gc, "collect", spy_collect)

    stream = PackedMicroBatches(
        tiny_config, MBS, SEQ_LEN, data_root="mmu", match_index="present"
    )
    batches = list(islice(iter(stream), 4))

    assert builds["n"] >= 2, "the stream was never rebuilt — error path not taken"
    assert collects["n"] >= 1, "rebuild did not reclaim the abandoned stream"
    assert len(batches) == 4  # recovered and kept producing valid batches
    for flat in batches:
        assert flat["input_ids"].shape == (MBS, SEQ_LEN)


def _span_order(sequencer, record, object_id):
    """Modality names in the order their spans appear in the sequence."""
    obj = sequencer.build({**record, "object_id": object_id}, epoch=0)
    starts = {
        name: int(mask.float().argmax())
        for name, mask in obj.masks.items()
        if mask.any()
    }
    return [name for name, _ in sorted(starts.items(), key=lambda kv: kv[1])]


def test_ar_replicas_reemit_each_record_under_a_different_span_order(
    tiny_config, tmp_path
):
    """Replay buys tokens without buying bytes — but only if the order changes.

    The corpus is transfer-bound, so re-emitting a record already in memory is
    nearly free. It is worth nothing if the replica repeats the same
    autoregressive factorisation, and it is unsafe if it breaks the
    exactly-once object_id_log audit. Both are checked here.
    """
    from astropt3.data.packing import ObjectSequencer
    from astropt3.data.synthetic import make_record

    log = tmp_path / "objects.log"
    stream = PackedMicroBatches(
        tiny_config, MBS, SEQ_LEN, object_id_log=log, ar_replicas=2
    )
    list(islice(iter(stream), 3))
    ids = Path(f"{log}.dp0").read_text().split()
    assert ids, "nothing logged"
    assert len(ids) == len(set(ids))  # audit stays one line per emitted sequence
    replicas = [i for i in ids if i.endswith("#1")]
    assert replicas, "ar_replicas=2 emitted no replicas"
    assert all(i.removesuffix("#1") in ids for i in replicas)  # paired with base

    # ar_replicas=1 is still the default, and emits no replicas at all
    plain = PackedMicroBatches(tiny_config, MBS, SEQ_LEN, ar_replicas=1)
    baseline = PackedMicroBatches(tiny_config, MBS, SEQ_LEN)
    a, b = next(iter(plain)), next(iter(baseline))
    assert torch.equal(a["input_ids"], b["input_ids"])

    # the mechanism: the span order is a pure function of (object_id, epoch),
    # so suffixing the id reseeds the ADR 0008 shuffle
    sequencer = ObjectSequencer(tiny_config)
    differed = sum(
        _span_order(sequencer, make_record(i), f"o{i}")
        != _span_order(sequencer, make_record(i), f"o{i}#1")
        for i in range(20)
    )
    assert differed, "replica ids produced identical span orders in 20 records"


def test_ar_replicas_rejects_a_nonsense_count(tiny_config):
    with pytest.raises(ValueError, match="ar_replicas must be >= 1"):
        PackedMicroBatches(tiny_config, MBS, SEQ_LEN, ar_replicas=0)


# -- ADR 0014 §7a: distinct-order enforcement ---------------------------------


def test_one_span_records_get_no_replica(tiny_config):
    """No alternative order exists, so a replica would be an exact duplicate.

    Identical duplicates would raise MFU and E_AR while training on nothing
    new — precisely the gaming §2 refuses to accept.
    """
    from astropt3.data.synthetic import make_record

    stream = PackedMicroBatches(tiny_config, 8, SEQ_LEN, ar_replicas=4)
    # a spectrum-only record with an unreliable Z carries a single span
    record = make_record(2, image_only_fraction=0.0, spectrum_only_fraction=1.0)
    record["ZWARN"] = 1  # gates the Z span out (ADR 0008)
    objects = stream._replica_objects(record)
    assert len(objects[0].order) == 1
    assert len(objects) == 1


def test_replicas_carry_distinct_span_orders(tiny_config):
    from astropt3.data.synthetic import make_record

    stream = PackedMicroBatches(tiny_config, 8, SEQ_LEN, ar_replicas=4)
    for i in range(12):
        objects = stream._replica_objects(make_record(i))
        orders = [obj.order for obj in objects]
        assert len(orders) == len(set(orders)), f"record {i} repeated an order"
        # replica 0 keeps the base id; replicas are suffixed for the audit
        assert "#" not in objects[0].object_id
        assert all("#" in obj.object_id for obj in objects[1:])


def test_replicas_are_capped_by_the_number_of_distinct_permutations(tiny_config):
    """A two-span record supports at most one extra ordering (2! = 2)."""
    from astropt3.data.synthetic import make_record

    stream = PackedMicroBatches(tiny_config, 8, SEQ_LEN, ar_replicas=8)
    record = make_record(2, image_only_fraction=0.0, spectrum_only_fraction=1.0)
    objects = stream._replica_objects(record)
    assert len(objects[0].order) == 2  # spectra + Z
    assert len(objects) == 2
    assert len({obj.order for obj in objects}) == 2


# -- ADR 0014 §7b: decorrelation ----------------------------------------------


def test_no_two_replicas_of_one_object_share_a_packed_row(tiny_config):
    """Document masking stops cross-attention, not gradient repetition."""
    from astropt3.data.synthetic import make_record

    stream = PackedMicroBatches(tiny_config, MBS, SEQ_LEN, ar_replicas=2)
    placements = []
    for i in range(8):
        objects = stream._replica_objects(make_record(i))
        if len(objects) > 1:
            placements.append(stream._place(objects, [0] * MBS))
    assert placements, "no record produced replicas"
    for chosen in placements:
        assert len(chosen) == len(set(chosen)), "two replicas landed in one row"


def test_placement_returns_none_when_no_distinct_rows_fit(tiny_config):
    from astropt3.data.synthetic import make_record

    stream = PackedMicroBatches(tiny_config, MBS, SEQ_LEN, ar_replicas=2)
    objects = stream._replica_objects(make_record(3))
    assert len(objects) == 2
    # both rows already too full to take another object -> close the batch
    full = [SEQ_LEN] * MBS
    assert stream._place(objects, full) is None
    # one row free is not enough for two replicas that must not share a row
    one_free = [0] + [SEQ_LEN] * (MBS - 1)
    assert stream._place(objects, one_free) is None


def test_placement_prefers_the_emptiest_row_and_is_deterministic(tiny_config):
    from astropt3.data.synthetic import make_record

    stream = PackedMicroBatches(tiny_config, MBS, SEQ_LEN, ar_replicas=1)
    objects = stream._replica_objects(make_record(1))
    used = [300, 10, 900] + [50] * (MBS - 3)
    first = stream._place(objects, used)
    assert first == [1]  # emptiest wins
    assert stream._place(objects, used) == first  # pure function of `used`


def test_more_replicas_than_rows_is_rejected(tiny_config):
    with pytest.raises(ValueError, match="exceeds micro_batch_size"):
        PackedMicroBatches(tiny_config, 2, SEQ_LEN, ar_replicas=3)


# -- ADR 0014 §5: sequence-assembly fingerprint -------------------------------


def _tag(config, **kwargs):
    kwargs.setdefault("ar_replicas", 1)
    return PackedMicroBatches(config, MBS, SEQ_LEN, **kwargs)._source_assembly


def test_fingerprint_separates_replica_counts(tiny_config):
    """The bug §5 exists to close: replicas 1 -> 2 passed the old check."""
    assert _tag(tiny_config, ar_replicas=1) != _tag(tiny_config, ar_replicas=2)


def test_fingerprint_separates_sequence_lengths(tiny_config):
    a = PackedMicroBatches(tiny_config, MBS, 512)._source_assembly
    b = PackedMicroBatches(tiny_config, MBS, 1024)._source_assembly
    assert a != b


def test_fingerprint_separates_tokenisation_policies(tiny_config):
    from astropt3.configuration_astropt3 import AstroPT3Config

    per_band = []
    for modality in tiny_config.modalities:
        modality = dict(modality)
        if modality["name"] == "images":
            modality.update(
                input_size=64,
                max_positions=432,
                channel_tokenization="per_band",
                band_order=["des-g", "des-r", "des-z"],
            )
        per_band.append(modality)
    other = AstroPT3Config(
        modalities=per_band,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
    )
    assert _tag(tiny_config) != _tag(other)


def test_fingerprint_keeps_the_assembly_readable_and_is_stable(tiny_config):
    tag = _tag(tiny_config)
    assert tag.startswith("synthetic:")  # assembly still legible at a glance
    assert tag == _tag(tiny_config)  # same policy -> same tag, run to run


def test_a_mismatched_fingerprint_rejects_the_stream_state(tiny_config):
    saved = PackedMicroBatches(tiny_config, MBS, SEQ_LEN, ar_replicas=1)
    state = saved.state_dict()
    resumed = PackedMicroBatches(tiny_config, MBS, SEQ_LEN, ar_replicas=2)
    with pytest.raises(ValueError, match="source_assembly"):
        resumed.load_state_dict(state)


def test_adjacent_placement_is_the_pre_decorrelation_behaviour(tiny_config):
    """The B1 arm exists to reproduce the measured result on the same footing."""
    from astropt3.data.synthetic import make_record

    stream = PackedMicroBatches(
        tiny_config, MBS, SEQ_LEN, ar_replicas=2, replica_placement="adjacent"
    )
    objects = stream._replica_objects(make_record(3))
    assert len(objects) == 2
    chosen = stream._place(objects, [0] * MBS)
    assert chosen == [0, 0]  # same row, as before §7b

    # and the fingerprint separates the two arms, so neither can resume onto
    # the other's stream state
    decorrelated = PackedMicroBatches(
        tiny_config, MBS, SEQ_LEN, ar_replicas=2, replica_placement="decorrelated"
    )
    assert stream._source_assembly != decorrelated._source_assembly


def test_replica_placement_rejects_an_unknown_mode(tiny_config):
    with pytest.raises(ValueError, match="replica_placement"):
        PackedMicroBatches(tiny_config, MBS, SEQ_LEN, replica_placement="sideways")


def test_adjacent_placement_allows_more_replicas_than_rows(tiny_config):
    """Only the decorrelated mode needs one row per replica."""
    PackedMicroBatches(
        tiny_config, 2, SEQ_LEN, ar_replicas=4, replica_placement="adjacent"
    )
