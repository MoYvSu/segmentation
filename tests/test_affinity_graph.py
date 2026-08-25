import numpy as np

from utils.affinity_graph import (
    DEFAULT_AFFINITY_OFFSETS,
    audit_instance_recovery,
    build_affinity_targets,
    reconstruct_affinity_components,
)


def test_oracle_affinity_recovers_nonconvex_instances_exactly():
    labels = np.zeros((12, 14), dtype=np.int32)
    labels[1:10, 1:3] = 1
    labels[8:10, 1:9] = 1
    labels[2:5, 8:12] = 2
    labels[5:8, 10:12] = 2
    affinity, valid = build_affinity_targets(labels)
    assert affinity.shape == (len(DEFAULT_AFFINITY_OFFSETS), 12, 14)
    assert np.all(affinity[~valid] == 0)
    prediction, graph_audit = reconstruct_affinity_components(
        labels > 0, affinity
    )
    recovery = audit_instance_recovery(labels, prediction)
    assert graph_audit["raw_component_count"] == 2
    assert recovery["exact_partition"]


def test_diagonal_affinity_connects_an_eight_connected_instance():
    labels = np.zeros((4, 4), dtype=np.int32)
    labels[1, 1] = 1
    labels[2, 2] = 1
    affinity, _ = build_affinity_targets(labels)
    prediction, _ = reconstruct_affinity_components(labels > 0, affinity)
    assert len(np.unique(prediction[prediction > 0])) == 1


def test_disconnected_label_is_reported_as_split():
    labels = np.zeros((12, 12), dtype=np.int32)
    labels[1:3, 1:3] = 1
    labels[9:11, 9:11] = 1
    affinity, _ = build_affinity_targets(labels)
    prediction, _ = reconstruct_affinity_components(labels > 0, affinity)
    recovery = audit_instance_recovery(labels, prediction)
    assert recovery["split_gt_instance_count"] == 1
    assert not recovery["exact_partition"]


def test_component_cap_never_exceeds_uint8_contract():
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[::2, ::2] = np.arange(1, 401).reshape(20, 20)
    affinity, _ = build_affinity_targets(labels)
    prediction, audit = reconstruct_affinity_components(
        labels > 0, affinity, max_instances=255
    )
    assert prediction.dtype == np.uint8
    assert int(prediction.max()) <= 255
    assert audit["kept_instance_count"] == 255
    assert audit["merged_components_for_cap"] == 145

def test_uncapped_diagnostic_preserves_every_raw_component():
    labels = np.zeros((40, 40), dtype=np.int32)
    labels[::2, ::2] = np.arange(1, 401).reshape(20, 20)
    affinity, _ = build_affinity_targets(labels)
    prediction, audit = reconstruct_affinity_components(
        labels > 0, affinity, max_instances=None
    )
    assert prediction.dtype == np.int32
    assert int(prediction.max()) == 400
    assert audit["kept_instance_count"] == 400
    assert audit["merged_components_for_cap"] == 0
