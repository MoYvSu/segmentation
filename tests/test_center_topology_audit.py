import numpy as np

from tools.audit_g1_center_generalization import instance_topology


def test_edt_center_stays_inside_when_geometric_centroid_is_outside():
    instance_map = np.zeros((64, 64), dtype=np.int32)
    yy, xx = np.indices(instance_map.shape)
    ring = ((yy - 32) ** 2 + (xx - 32) ** 2 <= 25 ** 2) & (
        (yy - 32) ** 2 + (xx - 32) ** 2 >= 15 ** 2
    )
    instance_map[ring] = 1
    row = instance_topology(instance_map, "ring.png")[0]
    assert not row["centroid_inside"]
    assert row["edt_center_inside"]
    assert row["star_visible_fraction"] < 1.0
