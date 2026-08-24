from train_offset_geometry_g1 import deterministic_split


def test_g1_split_is_deterministic_and_keeps_g0_samples_in_train():
    names = [f"train_{index:03d}" for index in range(32)]
    train_a, val_a = deterministic_split(
        names, val_fraction=0.2, seed=42,
        forced_train=["train_001", "train_002"],
    )
    train_b, val_b = deterministic_split(
        list(reversed(names)), val_fraction=0.2, seed=42,
        forced_train=["train_001", "train_002"],
    )
    assert (train_a, val_a) == (train_b, val_b)
    assert len(train_a) == 26
    assert len(val_a) == 6
    assert "train_001" in train_a and "train_002" in train_a
    assert set(train_a).isdisjoint(val_a)
