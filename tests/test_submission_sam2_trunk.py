# -*- coding: utf-8 -*-
"""交付包训练编码器的CPU假SAM2合同；不导入vendor或接触真实权重。"""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


SOURCE = Path(__file__).resolve().parents[1] / 'submission/AIC-2026-59388388-代码与数据/training/source/models/sam2_encoder.py'


@pytest.fixture
def fake_sam2(monkeypatch, tmp_path):
    h = SimpleNamespace(build_calls=[], loads=[], models=[])

    class Trunk(torch.nn.Linear):
        def __init__(self):
            super().__init__(2, 3)

        def load_state_dict(self, state, strict=True, **kwargs):
            h.loads.append(dict(keys=set(state), strict=strict))
            return super().load_state_dict(state, strict=strict, **kwargs)

    class Decoder(torch.nn.Linear):
        def __init__(self):
            super().__init__(2, 4)

        def load_state_dict(self, *args, **kwargs):
            raise AssertionError('Native decoder must never load pretrained weights')

        def _load_from_state_dict(self, *args, **kwargs):
            raise AssertionError('Native decoder must never participate in recursive state loading')

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.image_encoder = torch.nn.Module()
            self.image_encoder.trunk = Trunk()
            self.sam_mask_decoder = Decoder()

        def load_state_dict(self, *args, **kwargs):
            raise AssertionError('Whole official checkpoint must not be loaded into the model')

    def build_sam2(**kwargs):
        h.build_calls.append(kwargs)
        assert kwargs['ckpt_path'] is None
        model = Model().to(kwargs['device']).eval()
        h.models.append((model,{name:value.clone() for name,value in model.sam_mask_decoder.state_dict().items()}))
        return model

    package, module = ModuleType('sam2'), ModuleType('sam2.build_sam')
    package.__path__ = []
    module.build_sam2 = build_sam2
    monkeypatch.setitem(sys.modules,'sam2',package)
    monkeypatch.setitem(sys.modules,'sam2.build_sam',module)
    spec = importlib.util.spec_from_file_location('final_package_sam2_encoder',SOURCE)
    imported = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imported)
    h.encoder = imported.SAM2Encoder
    h.Model = Model
    h.checkpoint = tmp_path/'official.pt'
    h.state = {
        'image_encoder.trunk.weight':torch.full((3,2),.125),
        'image_encoder.trunk.bias':torch.tensor([.1,.2,.3]),
        'sam_mask_decoder.weight':torch.full((4,2),9876.),
        'sam_mask_decoder.bias':torch.full((4,),9876.),
        'image_encoder.neck.unused':torch.tensor([7654.])}
    torch.save({'model':h.state},h.checkpoint)
    return h


def test_only_prefix_filtered_trunk_strictly_loaded_and_decoder_sentinel_ignored(fake_sam2):
    h = fake_sam2
    encoder = h.encoder(config_file='dummy.yaml',ckpt_path=h.checkpoint,device='cpu',freeze=True)
    assert h.build_calls == [dict(config_file='dummy.yaml',ckpt_path=None,device='cpu',
                                 mode='eval',apply_postprocessing=False)]
    assert h.loads == [dict(keys={'weight','bias'},strict=True)]
    assert torch.equal(encoder.trunk.weight,h.state['image_encoder.trunk.weight'])
    assert torch.equal(encoder.trunk.bias,h.state['image_encoder.trunk.bias'])
    assert all(not parameter.requires_grad for parameter in encoder.trunk.parameters())
    assert encoder.trunk.training is False
    model, before = h.models[0]
    assert all(torch.equal(value,model.sam_mask_decoder.state_dict()[name]) for name,value in before.items())


def test_none_checkpoint_export_path_skips_torch_load_and_preserves_freeze_false(fake_sam2, monkeypatch):
    h = fake_sam2
    monkeypatch.setattr(torch,'load',lambda *args,**kwargs:pytest.fail('None checkpoint must not open weights'))
    encoder = h.encoder(ckpt_path=None,device='cpu',freeze=False)
    assert h.build_calls[0]['ckpt_path'] is None and h.loads == []
    assert encoder.trunk.training is False
    assert all(parameter.requires_grad for parameter in encoder.trunk.parameters())


def test_full_random_initialization_consumption_is_preserved(fake_sam2):
    h = fake_sam2
    torch.manual_seed(20261101)
    reference = h.Model().to('cpu').eval()
    expected_rng = torch.get_rng_state().clone()
    expected_next = torch.rand(8)
    torch.manual_seed(20261101)
    encoder = h.encoder(ckpt_path=h.checkpoint,device='cpu')
    assert torch.equal(torch.get_rng_state(),expected_rng)
    assert torch.equal(torch.rand(8),expected_next)
    assert encoder.trunk is h.models[-1][0].image_encoder.trunk
    assert reference.sam_mask_decoder.weight.shape == (4,2)


@pytest.mark.parametrize('defect',['missing_key','wrong_shape','no_trunk','not_official'])
def test_incompatible_trunk_or_checkpoint_format_fails_strictly(fake_sam2, defect):
    h = fake_sam2
    state = dict(h.state)
    if defect == 'missing_key': del state['image_encoder.trunk.bias']
    elif defect == 'wrong_shape': state['image_encoder.trunk.weight'] = torch.ones(1,1)
    elif defect == 'no_trunk': state = {'sam_mask_decoder.weight':torch.ones(4,2)}
    payload = state if defect == 'not_official' else {'model':state}
    torch.save(payload,h.checkpoint)
    with pytest.raises((RuntimeError,ValueError)):
        h.encoder(ckpt_path=h.checkpoint,device='cpu')
    assert h.build_calls[0]['ckpt_path'] is None
    assert all(row['strict'] is True and row['keys'] <= {'weight','bias'} for row in h.loads)
