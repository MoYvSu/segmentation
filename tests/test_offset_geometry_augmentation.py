import torch

from data.offset_geometry_augmentation import (
    augment_cached_geometry_sample,
    transform_offset_vectors,
)


def test_offset_vector_components_follow_rotation_and_flips():
    offsets = torch.zeros((2, 2, 3), dtype=torch.float32)
    offsets[0].fill_(2.0)
    offsets[1].fill_(3.0)
    rotated = transform_offset_vectors(offsets, False, False, 1)
    assert torch.all(rotated[0] == -3.0)
    assert torch.all(rotated[1] == 2.0)
    flipped = transform_offset_vectors(offsets, True, True, 0)
    assert torch.all(flipped[0] == -2.0)
    assert torch.all(flipped[1] == -3.0)


def test_rectangular_content_rotation_rebuilds_letterbox_padding():
    sample = {
        "image": torch.full((3, 8, 8), 0.5),
        "center_target": torch.zeros((1, 4, 4)),
        "offset_target": torch.zeros((2, 4, 4)),
        "foreground": torch.zeros((1, 4, 4), dtype=torch.bool),
        "valid_content": torch.zeros((1, 4, 4), dtype=torch.bool),
        "instance_map": torch.zeros((4, 4), dtype=torch.int64),
        "content_shape": torch.tensor([3, 4], dtype=torch.int32),
        "image_name": "synthetic.jpg",
    }
    sample["foreground"][:, :3, :4] = True
    sample["instance_map"][:3, :4] = 1
    transformed = augment_cached_geometry_sample(
        sample, hflip=False, vflip=False, rotation_k=1
    )
    assert transformed["content_shape"].tolist() == [4, 3]
    assert transformed["valid_content"][:, :4, :3].all()
    assert not transformed["valid_content"][:, :, 3:].any()
    assert transformed["foreground"].sum() == 12
    assert transformed["image"].shape == (3, 8, 8)
    assert float(transformed["image"][:, :, -1].mean()) > 0.0
