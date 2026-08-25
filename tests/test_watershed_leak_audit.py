import numpy as np

from tools.audit_watershed_leaks import close_boundary, hysteresis_boundary


def test_hysteresis_keeps_only_weak_component_attached_to_strong_boundary():
    probability = np.zeros((7, 9), dtype=np.float32)
    probability[1, 1:5] = 0.3
    probability[1, 1] = 0.8
    probability[5, 5:8] = 0.3
    output = hysteresis_boundary(probability, 0.25, 0.5)
    assert output[1, 1:5].all()
    assert not output[5, 5:8].any()


def test_close_boundary_repairs_one_pixel_gap():
    boundary = np.zeros((7, 7), dtype=np.uint8)
    boundary[3, 1:3] = 1
    boundary[3, 4:6] = 1
    closed = close_boundary(boundary, 1)
    assert closed[3, 3] == 1
