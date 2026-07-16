import torch

from astropt3.data.packing import PackedCollator
from astropt3.data.synthetic import record_stream
from astropt3.modeling_astropt3 import left_shift_mask
from astropt3.tokenization import BOS_ID, PAD_ID, modality_token_ids


def test_object_sequence_structure(sequencer, full_record):
    obj = sequencer.build(full_record)
    assert len(obj) == 180  # 1 bos + (1+144+1) images + (1+31+1) spectra
    begin_img, ph_img, end_img = modality_token_ids("images")
    begin_spec, ph_spec, end_spec = modality_token_ids("spectra")
    ids = obj.input_ids
    assert ids[0] == BOS_ID
    assert ids[1] == begin_img and ids[146] == end_img
    assert (ids[2:146] == ph_img).all()
    assert ids[147] == begin_spec and ids[179] == end_spec
    assert (ids[148:179] == ph_spec).all()
    assert obj.masks["images"].sum() == 144 and obj.values["images"].shape == (144, 192)
    assert obj.masks["spectra"].sum() == 31 and obj.values["spectra"].shape == (31, 256)
    # spectra positions are normalized wavelengths in ~[0, 1]
    pos = obj.positions["spectra"]
    assert pos.shape == (31, 1) and (pos > 0).all() and (pos < 1.1).all()


def test_image_only_object(sequencer, image_only_record):
    obj = sequencer.build(image_only_record)
    assert len(obj) == 147  # 1 bos + 146 images block
    assert "spectra" not in obj.masks


def test_collator_packs_whole_objects(sequencer, collator):
    objs = [sequencer.build(r) for r in record_stream(6)]
    batch = collator(objs)
    B, T = batch["input_ids"].shape
    assert T == 896
    # every object's position_ids run 0..len-1 contiguously: object starts are
    # exactly the positions where position_ids == 0 and the token is not pad
    for b in range(B):
        ids = batch["input_ids"][b]
        pos = batch["position_ids"][b]
        starts = [t for t in range(T) if pos[t] == 0 and ids[t] != PAD_ID]
        for t in starts:
            assert ids[t] == BOS_ID  # objects begin with <|bos|>, never split
    # flattened values align with row-major mask order
    for m, values in batch["modality_values"].items():
        assert batch["modality_masks"][m].sum().item() == len(values)


def test_values_row_major_alignment(sequencer, collator):
    objs = [sequencer.build(r) for r in record_stream(4)]
    batch = collator(objs)
    # rebuild the concatenation by scanning rows/objects in order and compare
    for m in batch["modality_values"]:
        expected = torch.cat([o.values[m] for o in objs if m in o.values], dim=0)
        assert torch.equal(batch["modality_values"][m], expected)


def test_left_shift_mask():
    mask = torch.tensor([[False, False, True, True, False]])
    shifted = left_shift_mask(mask)
    assert shifted.tolist() == [[False, True, True, False, False]]


def test_begin_token_predicts_first_patch(sequencer, collator, full_record):
    obj = sequencer.build(full_record)
    batch = collator([obj])
    begin_img, _, _ = modality_token_ids("images")
    shifted = left_shift_mask(batch["modality_masks"]["images"])
    begin_pos = (batch["input_ids"][0] == begin_img).nonzero()[0, 0]
    assert shifted[0, begin_pos]  # hidden at <|begin_images|> predicts patch 0
