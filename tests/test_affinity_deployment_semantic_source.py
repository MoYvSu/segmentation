import numpy as np
import torch

import utils.affinity_deployment as deployment


class _Encoder:
    def __init__(self):
        self.calls = 0

    def __call__(self, tensor):
        self.calls += 1
        return [tensor]


class _Reference:
    def __init__(self):
        self.encoder = _Encoder()
        self.forward_calls = 0

    def __call__(self, tensor):
        self.forward_calls += 1
        return torch.full((1, 2, 8, 10), -9.0)


class _Geometry:
    def __call__(self, features):
        return {"affinity_logits": torch.zeros((1, 8, 4, 5))}


class _System:
    def __init__(self):
        self.reference_model = _Reference()
        self.geometry_feature_adapter = None
        self.geometry_decoder = _Geometry()


def test_replace_reference_semantic_uses_challenger_only():
    system = _System()
    challenger_logits = torch.full((1, 1, 8, 10), 2.0)
    original_prepare = deployment.prepare_image
    deployment.prepare_image = lambda *_args, **_kwargs: (
        np.zeros((8, 10, 3), dtype=np.uint8),
        torch.zeros((1, 3, 8, 10)),
        0,
        0,
    )
    try:
        _, semantic, output, _, vote_logits = (
            deployment.predict_maps_with_challenger(
                system,
                lambda _features, _image: challenger_logits,
                "unused.jpg",
                10,
                torch.device("cpu"),
                replace_reference_semantic=True,
            )
        )
    finally:
        deployment.prepare_image = original_prepare

    assert system.reference_model.forward_calls == 0
    assert system.reference_model.encoder.calls == 1
    assert vote_logits is None
    assert torch.equal(semantic, challenger_logits)
    assert torch.equal(output[:, :1], challenger_logits)
