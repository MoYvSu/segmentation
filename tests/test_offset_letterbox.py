import numpy as np

from utils.offset_letterbox import (
    inverse_letterbox_instances,
    letterbox_instance_geometry,
)


def test_geometry_padding_is_zero_and_invalid():
    instances = np.ones((60, 100), dtype=np.int32)
    grid, valid, metadata = letterbox_instance_geometry(
        instances, input_size=1024, output_grid=256
    )
    assert metadata.content_width == 256
    assert metadata.content_height < 256
    assert np.all(grid[~valid] == 0)
    assert not np.any(valid[metadata.content_height :, :])


def test_inverse_letterbox_restores_shape_and_ids():
    instances = np.zeros((60, 100), dtype=np.int32)
    instances[5:30, 8:42] = 1
    instances[20:55, 55:95] = 2
    grid, _, metadata = letterbox_instance_geometry(
        instances, input_size=1024, output_grid=512
    )
    restored = inverse_letterbox_instances(grid, metadata)
    assert restored.shape == instances.shape
    assert set(np.unique(restored)) == {0, 1, 2}
