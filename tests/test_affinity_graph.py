import numpy as np

from utils.affinity_graph import (
    DEFAULT_AFFINITY_OFFSETS,
    audit_instance_recovery,
    build_affinity_targets,
    reconstruct_affinity_components,
    regularize_affinity_components,
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

def test_channel_specific_threshold_can_disable_long_edges():
    labels = np.zeros((6, 8), dtype=np.int32)
    labels[2, 1] = 1
    labels[2, 5] = 2
    affinity = np.zeros((len(DEFAULT_AFFINITY_OFFSETS), 6, 8), dtype=np.float32)
    affinity[6, 2, 1] = 0.9
    merged, _ = reconstruct_affinity_components(labels > 0, affinity, threshold=0.5)
    separated, _ = reconstruct_affinity_components(
        labels > 0, affinity,
        threshold=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.1, 1.1],
    )
    assert len(np.unique(merged[merged > 0])) == 1
    assert len(np.unique(separated[separated > 0])) == 2


def test_regularizer_reassigns_boundary_fragments_without_losing_pixels():
    components = np.zeros((7, 11), dtype=np.int32)
    components[:, :4] = 1
    components[:, 7:] = 2
    components[:, 4:7] = np.arange(21).reshape(7, 3) + 3
    partition, audit = regularize_affinity_components(
        components, min_component_area=5
    )
    assert set(np.unique(partition)) == {1, 2}
    assert np.all(partition > 0)
    assert audit["retained_core_count"] == 2
    assert audit["removed_fragment_count"] == 21
    assert audit["reassigned_pixels"] == 21
