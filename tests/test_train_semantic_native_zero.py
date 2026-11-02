# -*- coding: utf-8 -*-
"""训练器小CPU合同；不加载赛题图像、SAM2、D5a或CUDA。"""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from tools import train_semantic_native_zero as train


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(.3)
        self.linear = nn.Conv2d(3, 1, 1)

    def forward(self, features, image):
        assert features[0] is image
        return self.linear(self.dropout(features[0])) + image.mean(1, keepdim=True)*.05


class Backend(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 3, 1))
        self.semantic_decoder = Head()
        self.affinity_decoder = nn.Conv2d(3, 8, 1)
        self.register_buffer('frozen_buffer', torch.tensor([1.]))


class Restorer(nn.Module):
    num_steps = 16
    def __init__(self):
        super().__init__()
        self.register_buffer('schedule', torch.tensor([.03]))

    def q_sample(self, state, condition, step, noise):
        assert noise.dtype == torch.float32
        return state + noise

    def denoise(self, state, condition, step):
        return state*.9


def criterion(prediction, target, instance_map):
    loss = (prediction[:, 0]-target[:, 0]).square().mean()
    return loss, float(loss.detach()), 0.


def small_features(model, restorer, image):
    # 此处只隔离大模型/1024资源开销，保留真实decoder/dropout/optimizer/RNG恢复流程。
    return image, [image], dict(noise_nonzero=0, initial_equals_condition=True, q_sample_calls=1)


def sample():
    return dict(image=torch.arange(3*8*8).reshape(3, 8, 8).float()/192,
                semantic_target=torch.ones(1,8,8), semantic_boundary=torch.zeros(1,8,8),
                semantic_instance_map=torch.ones(8,8,dtype=torch.long))


def fixture_model():
    torch.manual_seed(123)
    model, restorer = Backend(), Restorer()
    frozen = train.freeze_contract(model, restorer, expected_parameters=4)
    optimizer = torch.optim.AdamW(model.semantic_decoder.parameters(), lr=3e-5, eps=1e-4,
                                 weight_decay=1e-4, foreach=False)
    return model, restorer, optimizer, frozen


def test_recipe_and_global_schedule():
    config = dict(semantic_native_zero=deepcopy(train.FIXED),
                  direct_semantic_affinity=dict(semantic_loss=train.LOSS))
    result = train.validate_config(config)
    assert result['epochs'] == 20
    assert train.learning_rate(1) == 1.5e-5
    assert train.learning_rate(2) == train.learning_rate(3) == 3e-5
    assert train.learning_rate(20) == pytest.approx(3e-6)
    config['semantic_native_zero']['restoration_noise'] = 'single'
    with pytest.raises(ValueError):
        train.validate_config(config)
    args = train.parse_args(['--config','c','--output','o','--backend','b','--restorer','r',
                             '--raw-dir','raw','--gt-dir','gt','--smoke-updates','3','--resume','last'])
    assert args.smoke_updates == 3


def test_real_zero_probe_forbids_rng_and_legacy_path():
    model = SimpleNamespace(encoder=lambda image: [image])
    restorer = Restorer()
    image = torch.ones(1,3,1024,1024)
    # 外层spy若被调用直接失败；zero分支不得生成随机噪声。
    with patch.object(torch, 'randn', side_effect=AssertionError('RNG used')) as spy:
        restored, features, audit = train.restored_features(model, restorer, image)
    assert spy.call_count == 0
    assert features[0] is restored
    assert audit['noise_nonzero'] == 0 and audit['initial_equals_condition']
    with train.no_restoration_rng(), pytest.raises(RuntimeError):
        train.backend_adaptation.restore_first(None, image, 0)


def test_only_semantic_trainable_buffers_frozen_and_rng_preserved():
    model, restorer, optimizer, frozen = fixture_model()
    train.train_mode(model)
    assert model.semantic_decoder.training and not model.encoder.training
    assert all(parameter.requires_grad == name.startswith('semantic_decoder.')
               for name, parameter in model.named_parameters())
    train.assert_frozen(model, restorer, frozen)
    state = train.rng_state()
    with train.preserve_rng():
        torch.rand(20)
        np.random.rand(10)
    assert torch.equal(train.rng_state()['torch'], state['torch'])
    model.frozen_buffer.add_(1)
    with pytest.raises(RuntimeError, match='冻结backend'):
        train.assert_frozen(model, restorer, frozen)


def assert_optimizer_exact(left, right):
    assert left['param_groups'] == right['param_groups']
    for key in left['state']:
        for field, value in left['state'][key].items():
            other = right['state'][key][field]
            if torch.is_tensor(value):
                assert torch.equal(value, other)
            else:
                assert value == other


def test_head_only_checkpoint_exact_next_update_resume(tmp_path):
    torch.set_num_threads(1)
    model, restorer, optimizer, frozen = fixture_model()
    identity = dict(base_backend_file_sha256=train.BACKEND_SHA,
                    base_semantic_sha256=train.tensor_sha(model.semantic_decoder.state_dict().items()),
                    semantic_architecture={'test': 'toy'}, frozen=frozen)
    train.train_mode(model)
    with patch.object(train, 'restored_features', new=small_features):
        train.semantic_update(model, restorer, criterion, optimizer, sample(), torch.device('cpu'))
        with patch.object(train, 'ensure_space', new=lambda *args, **kwargs: None):
            train.save_delta(tmp_path/'last.pt', model, optimizer, identity,
                             {'completed_updates':1}, output_dir=tmp_path)
        reference = train.semantic_update(model, restorer, criterion, optimizer, sample(), torch.device('cpu'))
    reference_head = deepcopy(model.semantic_decoder.state_dict())
    reference_optimizer = deepcopy(optimizer.state_dict())
    reference_rng = train.rng_state()
    resumed, other_restorer, other_optimizer, _ = fixture_model()
    checkpoint = torch.load(tmp_path/'last.pt', weights_only=False)
    assert not any(key.startswith(('encoder.', 'affinity_decoder.')) for key in checkpoint['semantic_decoder'])
    receipt = train.load_head_delta(resumed, tmp_path/'last.pt')
    assert receipt['strict_load'] and receipt['completed_updates'] == 1
    assert receipt['nonsemantic_before_sha256'] == receipt['nonsemantic_after_sha256']
    other_optimizer.load_state_dict(checkpoint['optimizer'])
    train.restore_rng(checkpoint['rng'])
    train.train_mode(resumed)
    with patch.object(train, 'restored_features', new=small_features):
        replay = train.semantic_update(resumed, other_restorer, criterion, other_optimizer,
                                       sample(), torch.device('cpu'))
    assert replay == reference
    for key, value in reference_head.items():
        assert torch.equal(value, resumed.semantic_decoder.state_dict()[key])
    assert_optimizer_exact(reference_optimizer, other_optimizer.state_dict())
    assert torch.equal(reference_rng['torch'], train.rng_state()['torch'])
    train.assert_frozen(resumed, other_restorer, frozen)
    resumed.frozen_buffer.add_(1)
    with pytest.raises(ValueError, match='基座'):
        train.load_head_delta(resumed, tmp_path/'last.pt')
