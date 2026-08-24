import numpy as np

from data.offset_geometry_dataset import adaptive_center_heatmap


def test_adaptive_heatmap_has_one_exact_peak_per_instance():
    instances = np.zeros((64, 96), dtype=np.int32)
    instances[5:30, 6:42] = 1
    instances[28:60, 54:92] = 2
    heatmap = adaptive_center_heatmap(instances)
    assert heatmap.shape == instances.shape
    assert int(np.sum(heatmap >= 1.0 - 1e-6)) == 2
    assert float(heatmap.min()) >= 0.0
    assert float(heatmap.max()) == 1.0
