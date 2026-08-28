import numpy as np

from utils.post_process import classify_instance_partition


def test_partition_assigns_every_pixel_and_obeys_cap():
    regions = np.arange(20 * 20, dtype=np.int32).reshape(20, 20) + 1
    probability = np.full((20, 20), 0.75, dtype=np.float32)
    instances, classes, audit = classify_instance_partition(
        regions, probability, min_area=2, max_instance_id=17
    )
    assert instances.dtype == np.uint8
    assert np.all(instances > 0)
    assert int(instances.max()) <= 17
    assert len(classes) <= 17
    assert audit["unassigned_pixels"] == 0


def test_partition_uses_supplied_semantic_probability():
    regions = np.ones((8, 12), dtype=np.int32)
    regions[:, 6:] = 2
    probability = np.zeros((8, 12), dtype=np.float32)
    probability[:, :6] = 0.9
    instances, classes, _ = classify_instance_partition(
        regions, probability, min_area=1, max_instance_id=255
    )
    left_id = int(instances[2, 2])
    right_id = int(instances[2, 9])
    assert classes[left_id] == 1
    assert classes[right_id] == 0
