import torch

from data.affinity_geometry_augmentation import crop_affinity_sample


def test_affinity_crop_resizes_image_and_instances_consistently():
    instances = torch.zeros((8, 8), dtype=torch.int64)
    instances[2:6, 2:6] = 7
    sample = {
        "image": torch.zeros((3, 16, 16), dtype=torch.float32),
        "instance_map": instances,
        "foreground": (instances > 0).unsqueeze(0),
        "valid_content": torch.ones((1, 8, 8), dtype=torch.bool),
        "center_target": torch.zeros((1, 8, 8)),
        "offset_target": torch.zeros((2, 8, 8)),
        "content_shape": torch.tensor([8, 8], dtype=torch.int32),
        "input_content_shape": torch.tensor([16, 16], dtype=torch.int32),
    }
    result = crop_affinity_sample(sample, 0.5, 0.5, 0.5)
    assert result["image"].shape == (3, 16, 16)
    assert result["instance_map"].shape == (8, 8)
    assert result["valid_content"].all()
    assert set(torch.unique(result["instance_map"]).tolist()) <= {0, 7}
