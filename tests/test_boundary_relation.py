# -*- coding: utf-8 -*-
"""关系头的独立性、成对对称、共同随机初态和可续训合同。"""
import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_relation_trainer_test', ROOT/'train_boundary_relation.py')
trainer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = trainer
spec.loader.exec_module(trainer)


@pytest.fixture(autouse=True)
def restore_runtime():
    rng = trainer.capture_rng()
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    trainer.restore_rng(rng)
    torch.set_num_threads(threads)


@pytest.fixture
def relation_data(tmp_path):
    data_module = trainer.data_module()
    h, count = 256, 4
    generator = np.random.default_rng(4)
    rgb = generator.integers(30, 230, size=(count, 3, h, h), dtype=np.uint8)
    maps = np.zeros((count, 2, h, h), np.float16)
    maps[:, :, :, 126:130] = .8
    masks = np.zeros((count, 4, h, h), np.uint8)
    masks[:, 0, :, :125] = 1
    masks[:, 1, :, 131:] = 1
    masks[:, 2, :, 126:130] = 1
    masks[:, 3] = 1
    manifest = dict(format=data_module.FORMAT, complete=True, patch_size=h, shards=[], monitor_shards=[],
                    monitor=dict(train_sample_ids=['train-0', 'train-1']))
    for split, n, source in [('train', count, 'train_synthetic'), ('monitor', 1, 'unlabeled_synthetic')]:
        path = tmp_path/f'{split}.npz'
        labels = np.array([0, 1, 0, 1], np.uint8) if split == 'train' else np.array([255], np.uint8)
        np.savez(path, rgb=rgb[:n], maps=maps[:n], masks=masks[:n], labels=labels)
        entry = dict(path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), count=n, source=source,
                     samples=[dict(sample_id=f'{split}-{i}', source=source, label=int(labels[i]), reason='synthetic_test_only') for i in range(n)])
        manifest['shards' if split == 'train' else 'monitor_shards'].append(entry)
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps(manifest), encoding='utf8')
    cfg = dict(seed=11, epochs=5, batch_size=2, num_workers=0, precision='float32', apply_predicted_merges=False,
               no_validation_holdout=True, learning_rate=3e-4, minimum_learning_rate=3e-5,
               weight_decay=1e-4, max_ram_gib=1, augment_d4=True, monitor_epochs=list(range(6)))
    cfg = trainer.validated_recipe(dict(boundary_relation=cfg))
    return path, cfg


def data_for(path, mode):
    cls = trainer.data_module().BoundaryRelationDataset
    return (cls(path, mode=mode, seed=11, max_ram_gib=1),
            cls(path, mode=mode, seed=11, max_ram_gib=1, split='monitor', augment=False))


def new_model():
    return trainer.BoundaryRelationNet(widths=(8, 16, 24), hidden=16, seed=11)


def test_constructor_and_strict_load_leave_random_streams(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda *args, **kwargs: calls.append(args))
    before = torch.get_rng_state().clone()
    model = new_model()
    assert torch.equal(before, torch.get_rng_state()) and not calls
    initial = tmp_path/'initial.pt'
    trainer.create_initial(initial, model)
    restored = trainer.load_initial(initial, model.architecture)
    assert trainer.state_sha256(model) == trainer.state_sha256(restored)
    trained_shape = tmp_path/'model.pt'
    torch.save(dict(format=trainer.FORMAT, architecture=model.architecture, model_state_dict=model.state_dict(),
                    initial_state_sha256=model.initial_state_sha256, model_sha256=model.initial_state_sha256), trained_shape)
    trainer._model.load_boundary_relation(trained_shape)
    assert torch.equal(before, torch.get_rng_state()) and not calls


def test_exact_side_symmetry_without_sam_or_rgb_mask_features(relation_data):
    path, _ = relation_data
    data, _ = data_for(path, 'rgb')
    batch = trainer.data_module().collate_boundary_relation([data[0], data[1]])
    model = new_model().eval()
    model.validate_inputs(batch['input'], batch['masks'], check_values=True)
    before = trainer.state_sha256(model)
    with torch.no_grad():
        a = model(batch['input'], batch['masks'])
        b = model(batch['input'], batch['masks'][:, [1, 0, 2, 3]])
    assert a.shape == (2,) and torch.equal(a, b)
    assert trainer.state_sha256(model) == before
    assert not any(isinstance(module, (torch.nn.BatchNorm2d, torch.nn.Dropout)) for module in model.modules())
    assert model.encoder[0].in_channels == 5


def test_initial_rejects_faked_trained_state(tmp_path):
    model = new_model()
    initial = tmp_path/'initial.pt'
    trainer.create_initial(initial, model)
    payload = torch.load(initial, weights_only=False)
    payload['model_state_dict'][next(iter(payload['model_state_dict']))].view(-1)[0] += .01
    fake = tmp_path/'fake.pt'
    torch.save(payload, fake)
    with pytest.raises(ValueError):
        trainer.load_initial(fake, model.architecture)


def test_inference_load_accepts_cross_platform_initial_rounding_but_rejects_corruption(tmp_path, relation_data):
    model = new_model().eval()
    local_initial = model.initial_state_sha256
    recorded_initial = '2569a361ee6810071511df0911dfe6a517a80a3f24c8bb2d9e9c337fcb0e4da0'
    assert recorded_initial != local_initial
    with torch.no_grad():
        model.head[-1].bias.add_(.25)  # 自洽的已训练状态，不再假设等于任一平台的随机e0。
    identity = dict(format=trainer.FORMAT, architecture=model.architecture, initial_state_sha256=recorded_initial)
    payload = dict(format=trainer.FORMAT, architecture=model.architecture,
                   initial_state_sha256=recorded_initial, model_state_dict=model.state_dict(),
                   model_sha256=trainer.state_sha256(model), identity=identity,
                   identity_sha256=trainer.canonical_sha(identity))
    path = tmp_path/'portable.pt'
    torch.save(payload, path)
    restored = trainer._model.load_boundary_relation(path)
    assert trainer.state_sha256(restored) == trainer.state_sha256(model)
    assert restored.initial_state_sha256 == recorded_initial
    assert restored.constructor_initial_state_sha256 == local_initial
    data, _ = data_for(relation_data[0], 'rgb')
    sample = data[0]
    with torch.no_grad():
        assert torch.equal(model(sample['input'][None], sample['masks'][None]),
                           restored(sample['input'][None], sample['masks'][None]))
    invalid = []
    bad = deepcopy(payload); bad['model_sha256'] = '0'*64; invalid.append(bad)
    bad = deepcopy(payload); bad['architecture']['normalization'] = 'BatchNorm'; invalid.append(bad)
    bad = deepcopy(payload); bad['initial_state_sha256'] = 'not-a-digest'; invalid.append(bad)
    bad = deepcopy(payload); bad['identity']['initial_state_sha256'] = 'f'*64; invalid.append(bad)
    bad = deepcopy(payload); bad['identity_sha256'] = '0'*64; invalid.append(bad)
    for index, bad in enumerate(invalid):
        target = tmp_path/f'invalid_{index}.pt'
        torch.save(bad, target)
        with pytest.raises((ValueError, RuntimeError)):
            trainer._model.load_boundary_relation(target)


def test_monitor_preserves_rng_weights_and_all_modes():
    model = new_model().train()
    model.encoder[0].eval()
    modes = [module.training for module in model.modules()]
    state = trainer.capture_rng()
    before = trainer.state_sha256(model)
    with trainer.monitor_state(model):
        random.random()
        np.random.random()
        torch.rand(4)
    assert modes == [module.training for module in model.modules()]
    assert trainer.state_sha256(model) == before
    assert random.getstate() == state['python']
    assert np.array_equal(np.random.get_state()[1], state['numpy'][1])
    assert torch.equal(torch.get_rng_state(), state['torch'])


def test_real_smoke_resume_and_unknown_monitor_exclusion(tmp_path, relation_data):
    path, cfg = relation_data
    data, monitor = data_for(path, 'maps')
    initial = tmp_path/'initial.pt'
    model = new_model()
    trainer.create_initial(initial, model)
    initial_sha = trainer.file_sha(initial)
    out = tmp_path/'smoke'
    trainer.smoke(model, data, monitor, dict(boundary_relation=cfg), cfg, out, 'maps', initial)
    receipt = json.loads((out/'smoke.json').read_text())
    assert receipt['resumed_next_update_exact'] and receipt['strict_reload_prediction_exact']
    assert receipt['AB_swap_exact'] and receipt['optimizer_updates'] == 2
    assert receipt['constructors_and_strict_load_preserve_RNG']
    assert trainer.file_sha(initial) == initial_sha
    rows = json.loads((out/'monitor/e000/receipt.json').read_text())['rows']
    assert any(row['kind'] == 'unlabeled_no_GT' and row['label'] is None for row in rows)
    identity = trainer.build_identity(model, data, cfg, 'maps', initial)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'])
    with pytest.raises(ValueError):
        trainer.load_resume(out/'smoke_resume.pt', model, optimizer, identity, cfg, 2)
    unknown = trainer.data_module().collate_boundary_relation([monitor[0]])
    before = trainer.state_sha256(model)
    with pytest.raises(ValueError):
        trainer.optimization_step(model, optimizer, unknown, pos_weight=1., lr=3e-4, mode='maps')
    assert trainer.state_sha256(model) == before


def test_two_arms_share_initial_order_and_draw_hashes(tmp_path, relation_data):
    path, cfg = relation_data
    initial = tmp_path/'initial.pt'
    trainer.create_initial(initial, new_model())
    logs = []
    final_hashes = {}
    for mode in ('maps', 'rgb'):
        data, monitor = data_for(path, mode)
        model = trainer.load_initial(initial, new_model().architecture)
        out = tmp_path/mode
        trainer.train(model, data, monitor, dict(boundary_relation=cfg), cfg, out, mode, initial)
        rows = [json.loads(line) for line in (out/'steps.jsonl').read_text().splitlines()]
        logs.append(rows)
        assert len(rows) == 10 and all((out/f'monitor/e{e:03d}/receipt.json').is_file() for e in range(6))
        loaded = trainer._model.load_boundary_relation(out/'e005.pt')
        assert trainer.state_sha256(loaded) == trainer.state_sha256(model)
        final_hashes[mode] = trainer.state_sha256(model)
    assert [row['sample_ids'] for row in logs[0]] == [row['sample_ids'] for row in logs[1]]
    assert [row['shared_draw_sha256'] for row in logs[0]] == [row['shared_draw_sha256'] for row in logs[1]]
    assert [row['learning_rate'] for row in logs[0]] == [row['learning_rate'] for row in logs[1]]
    # 从e1检查点重建采样游标，接续四轮应与不中断训练的模型及draw序列完全相同。
    data, monitor = data_for(path, 'maps')
    model = trainer.load_initial(initial, new_model().architecture)
    resumed = tmp_path/'resumed_maps'
    trainer.train(model, data, monitor, dict(boundary_relation=cfg), cfg, resumed, 'maps', initial,
                  resume=tmp_path/'maps/e001.pt')
    assert trainer.state_sha256(model) == final_hashes['maps']
    rows = [json.loads(line) for line in (resumed/'steps.jsonl').read_text().splitlines()]
    assert [row['shared_draw_sha256'] for row in rows] == [row['shared_draw_sha256'] for row in logs[0][2:]]
