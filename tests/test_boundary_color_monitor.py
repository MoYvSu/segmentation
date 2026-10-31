# -*- coding: utf-8 -*-
"""RGB＋擦断＋ridge的真实e0过程图入口；仅使用合成CPU数据。"""
from copy import deepcopy
import hashlib
import json

import numpy as np
from PIL import Image
import pytest
import torch

import tools.boundary_rgb_monitor as monitor
from models.boundary_rgb import BoundaryRGBNAFNet
from test_boundary_rgb_erasure import batch as erasure_batch
from test_boundary_rgb_monitor import frozen_config
from test_boundary_rgb_restored import _restored
from tools.boundary_restore_readout import evaluate_frozen_readout
from utils.config import load_config


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_real_NAF_color_e0_monitor_pairs_and_restored_full_readout(tmp_path, monkeypatch):
    """不替换NAF前向、分块、RGB加载或最终读出，防止仅短测通过而e0崩溃。"""
    sample = erasure_batch()
    sample = {key: value[0] if torch.is_tensor(value) else value
              for key, value in sample.items()}
    sample['meta'] = sample['meta'][0]
    sample['source'] = 'train_001'
    sample['valid'][..., :4, :] = 0
    sample['ridge_instances'] = torch.ones_like(sample['target'], dtype=torch.int64)
    sample['ridge_instances'][..., 48:] = 2
    original_sample = deepcopy(sample)

    # 测试整图不带GT；只提供合成的冻结上游缓存和无损恢复RGB清单。
    height, width = 64, 80
    raw = np.full((height, width, 3), 27, np.uint8)
    boundary = np.full((height, width), .03, np.float32)
    boundary[:, 38:43] = .9
    semantic = np.full(boundary.shape, .8, np.float32)
    raw_probability = np.full(boundary.shape, .8, np.float32)
    frozen = frozen_config()
    ids, classes, _ = evaluate_frozen_readout(semantic, raw_probability, boundary, frozen)
    control = tmp_path / 'control.npz'
    np.savez_compressed(control, boundary=boundary, rgb=raw, semantic=semantic,
                        raw_probability=raw_probability, instances=ids)
    class_path = tmp_path / 'classes.json'
    class_path.write_text(json.dumps(classes), encoding='utf-8')
    rgb_dir = tmp_path / 'restored'
    rgb_dir.mkdir()
    rgb_manifest, _, restored = _restored(rgb_dir, raw, domain='monitor_test', name='test_101')
    restored_before = restored.copy()
    row = dict(source='test_101', path=str(control), control_classes=str(class_path),
               sha256=hashlib.sha256(control.read_bytes()).hexdigest())
    monkeypatch.setattr(monitor, '_fixed_controls', lambda *_: [(row, frozen)])
    monkeypatch.setattr(monitor, 'original_view', lambda *_: np.ones((96, 96), bool))

    config = load_config('config/train/boundary_color.yaml')
    cfg = config['boundary_naf']
    assert cfg['erasure']['enabled'] and cfg['ridge_supervision']['enabled']
    assert cfg['ridge_supervision']['weight'] == .1 and cfg['training']['adversarial_weight'] == 0
    cfg['monitor'].update(pair_indices=[0], erasure_pair_indices=[1],
        source_names=['test_101'], full_readout_steps=[0], thumbnail_width=64,
        roi_spec=None, restored_rgb_manifest=str(rgb_manifest))
    cfg['inference'].update(tile_size=32, overlap=8, reflect_pad=3, batch_size=4)
    before_config = deepcopy(config)
    model = BoundaryRGBNAFNet(width=4, encoder_blocks=(1,), middle_blocks=1,
        decoder_blocks=(1,), input_mode='rgb_boundary', upsampling='project_bilinear_v1').train()
    model.encoders[0].eval()  # 保留混合train/eval模式，monitor结束必须恢复。
    modes = [child.training for child in model.modules()]
    before_state = deepcopy(model.state_dict())
    inputs = []
    hook = model.register_forward_pre_hook(lambda _, args: inputs.append(args[0].detach().clone()))
    try:
        result = monitor.save_monitor(model, config, tmp_path / 'run', 0, {0: sample, 1: sample})
    finally:
        hook.remove()

    assert result['full_readout'] and result['erasure_training']
    assert result['input_mode'] == 'rgb_boundary' and len(result['rows']) == 3
    assert [item['monitor_group'] for item in result['rows'][:2]] == ['common_control', 'erasure_extra']
    assert [child.training for child in model.modules()] == modes
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before_state.items())
    assert config == before_config

    # 两个pair均真正走受损/干净前向；仅T不同，RGB不得被GT/显示灰mask改写。
    assert len(inputs) > 4 and all(value.shape[1] == 4 for value in inputs)
    for dirty, clean in ((inputs[0], inputs[1]), (inputs[2], inputs[3])):
        assert torch.equal(dirty[:, :3], sample['rgb'][None])
        assert torch.equal(clean[:, :3], sample['rgb'][None])
        assert torch.equal(dirty[:, 3:], sample['input_boundary'][None])
        assert torch.equal(clean[:, 3:], sample['clean_input_boundary'][None])
        assert not torch.equal(dirty[:, 3:], clean[:, 3:])
    assert all(torch.equal(value, sample[key]) for key, value in original_sample.items()
               if torch.is_tensor(value))

    # 全图真实tiler使用float32恢复RGB，没有fallback raw、重复/255或uint8量化。
    padded = np.pad(restored, ((3, 3), (3, 3), (0, 0)), mode='reflect')
    np.testing.assert_array_equal(inputs[4][0, :3].numpy(), padded[:32, :32].transpose(2, 0, 1))
    assert not np.array_equal(padded[:32, :32], np.rint(padded[:32, :32] * 255) / 255)
    last = result['rows'][-1]
    assert last['kind'] == 'unlabelled_fixed_monitor' and last['reference_final_exact']
    assert last['seeds_and_barriers_regenerated'] and not last['old_markers_reused']
    assert last['rgb_conditioning']['domain'] == 'monitor_test'
    assert last['inference']['input_rgb_dtype'] == 'float32'
    assert last['inference']['input_rgb_sha256'] == hashlib.sha256(restored.tobytes()).hexdigest()
    np.testing.assert_array_equal(np.load(rgb_dir / 'restored.npy'), restored_before)
    folder = tmp_path / 'run/monitor/s000000'
    assert Image.open(folder / 'pair_000.jpg').size[0] == 9 * 64
    for name in ('pair_000_local.jpg', 'pair_001.jpg', 'pair_001_local.jpg',
                 'test_101_whole.jpg', 'test_101_adapter.jpg', 'summary.json'):
        assert (folder / name).is_file(), name
    stored = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    assert stored['global_step'] == 0 and stored['full_readout']
