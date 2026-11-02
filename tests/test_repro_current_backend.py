# -*- coding: utf-8 -*-
"""后期编排的 CPU 合同；不用赛题图像、GT、大模型或 CUDA。"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from repro import current_backend as current


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def context(tmp_path):
    assets = {}
    for key in ('raw_dir', 'unlabeled_dir', 'purified_gt_dir', 'completed_gt_dir',
                'sam2_geometry_dir', 'sam2_source_dir', 'sam2_repo'):
        folder = tmp_path / key
        folder.mkdir()
        assets[key] = str(folder)
    assets['monitor_dir'] = assets['raw_dir']
    for index in range(4):
        Path(assets['raw_dir'], f'train_{index:03d}.jpg').write_bytes(b'not-loaded')
        Path(assets['raw_dir'], f'train_{index:03d}.json').write_text('{}', encoding='utf-8')
    for key in ('completed_gt_manifest', 'sam2_authorization_doc', 'backbone'):
        path = tmp_path / (key + '.json')
        path.write_text('{}', encoding='utf-8')
        assets[key] = str(path)
    Path(assets['sam2_geometry_dir'], 'manifest.jsonl').write_text('{}\n', encoding='utf-8')
    run = tmp_path / 'run'
    artifacts = {key: str(run / key / 'final.pt') for key in
                 ('joint', 'semantic', 'affinity', 'r1')}
    artifacts.update(d5a=str(run / 'd5a' / 'epoch_060.pt'),
                     backend=str(run / 'export_backend' / 'model' / 'backend.pt'),
                     restorer=str(run / 'export_backend' / 'model' / 'restorer.pt'))
    for key in ('joint', 'semantic', 'affinity', 'd5a'):
        path = Path(artifacts[key])
        path.parent.mkdir(parents=True)
        path.write_bytes(key.encode())
    (run / 'configs').mkdir()
    (run / 'configs' / 'd5a.yaml').write_text('rgb_restoration: {}\n', encoding='utf-8')
    return dict(format=current.CONTEXT_FORMAT, project_root=str(ROOT), run_dir=str(run),
                settings={'stages': {}}, assets=assets, artifacts=artifacts)


def test_fresh_r1_recipe_rebinds_every_source_without_legacy_audits(context):
    config = current.prepare_r1_config(context)
    assert config['affinity_geometry_g1']['reference_checkpoint'] == context['artifacts']['joint']
    assert config['affinity_geometry_g1']['geometry_init_checkpoint'] == context['artifacts']['affinity']
    assert config['semantic_challenger']['checkpoint'] == context['artifacts']['semantic']
    cfg = config['backend_adaptation']
    assert cfg['restoration']['checkpoint'] == context['artifacts']['d5a']
    assert cfg['restoration']['checkpoint_sha256'] == current._sha(context['artifacts']['d5a'])
    assert cfg['sam2_geometry']['manifest_sha256'] == current._sha(
        Path(context['assets']['sam2_geometry_dir']) / 'manifest.jsonl')
    assert not {'reused_control', 'control_sha256', 'penalty_reference', 'penalty_sha256',
                'reused_native', 'native_sha256'} & set(config['affinity_native'])
    assert 'affinity_connectivity' not in config
    assert 'affinity_interface' not in config
    assert 'baseline_prediction_dir' not in cfg
    assert config['inference']['test_dir'] == context['assets']['raw_dir']
    assert cfg['monitor']['images'] == ['train_000', 'train_001']
    assert cfg['epochs'] == 20 and cfg['manual_repeats'] == 2
    assert config['direct_semantic_affinity']['affinity_loss']['negative_weight'] == 1.


def test_fresh_r1_rejects_recipe_budget_changes(context):
    context['settings']['stages']['r1'] = {'overrides': {'backend_adaptation': {'epochs': 1}}}
    with pytest.raises(ValueError, match='20 epochs'):
        current.prepare_r1_config(context)


def test_fresh_paths_cannot_be_overridden_by_stale_config(context):
    context['settings']['stages']['r1'] = {'overrides': {
        'affinity_geometry_g1': {'reference_checkpoint': 'old.pt'},
        'backend_adaptation': {'restoration': {'checkpoint_sha256': '0' * 64}}}}
    config = current.prepare_r1_config(context)
    assert config['affinity_geometry_g1']['reference_checkpoint'] == context['artifacts']['joint']
    assert config['backend_adaptation']['restoration']['checkpoint_sha256'] != '0' * 64


def test_source_process_monitor_cannot_silently_use_test_images(context, tmp_path):
    other = tmp_path / 'test'
    other.mkdir()
    context['assets']['monitor_dir'] = str(other)
    with pytest.raises(ValueError, match='training-source directory'):
        current.prepare_r1_config(context)


def test_d5a_recipe_is_one_fresh_candidate_with_training_source_monitor(context):
    config = current.prepare_d5a_config(context)
    cfg = config['rgb_restoration']
    assert cfg['seed'] == 42
    assert cfg['data_dir'] == context['assets']['unlabeled_dir']
    assert cfg['monitor']['real']['data_dir'] == context['assets']['raw_dir']
    assert cfg['train']['epochs'] == 60
    assert cfg['model']['kappa'] == .03
    assert cfg['degradation']['spatial_blur']['endpoint_transition']['enabled']


def test_sampler_plan_matches_actual_dataloader_rng_consumption():
    for epoch in (1, 2, 20):
        for kind in (0, 1):
            seed = 20260928 + epoch * 2 + kind
            expected = [int(t.item()) for t in DataLoader(range(64), batch_size=1, shuffle=True,
                        num_workers=0, generator=torch.Generator().manual_seed(seed))]
            assert current.sampler_order(64, seed) == expected
    assert current.sampler_order(64, 9) != current.sampler_order(64, 10)


class TinyView(torch.utils.data.Dataset):
    def __init__(self, size):
        self.size, self.epoch = size, 0

    def __len__(self):
        return 64

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        return dict(draw_index=int(index), native_crop_size=self.size)


def tiny_mixed(kind, loader_seed):
    # 不构造真实数据集，直接给包装器装入两个可观察的读图替身。
    dataset = current.FreshMixedViewDataset.__new__(current.FreshMixedViewDataset)
    dataset.whole, dataset.native = TinyView(0), TinyView(1024)
    dataset.kind, dataset.options, dataset.loader_seed = kind, {'mixed_view_seed': 20260930}, loader_seed
    dataset.mapping, dataset.draw_order, dataset.plan = None, None, None
    return dataset


def test_fresh_view_mapping_pairs_different_source_orders_and_covers_all_draws():
    manual, pseudo = tiny_mixed('manual', 20260928), tiny_mixed('pseudo', 20260929)
    for epoch in (1, 2, 20):
        manual.set_epoch(epoch)
        pseudo.set_epoch(epoch)
        assert manual.plan == pseudo.plan
        assert manual.plan.count(0) == manual.plan.count(1024) == 32
        assert set(manual.mapping) == set(pseudo.mapping) == set(range(64))
        assert manual.draw_order != pseudo.draw_order
        for index, (a, b) in enumerate(zip(manual.draw_order, pseudo.draw_order)):
            assert manual[a]['training_view'] == pseudo[b]['training_view']
            assert manual[a]['native_crop_size'] == manual.plan[index]


class Affinity(nn.Module):
    def __init__(self):
        super().__init__()
        self.output = nn.Conv2d(3, 1, 1)

    def forward(self, image):
        return {'affinity_logits': self.output(image)}


class Backend(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Conv2d(3, 3, 1)
        self.semantic_decoder = nn.Conv2d(3, 1, 1)
        self.affinity_decoder = Affinity()


def test_one_r1_update_steps_once_and_changes_only_affinity():
    model, restorer = Backend(), nn.Conv2d(3, 3, 1)
    model.requires_grad_(False)
    model.affinity_decoder.requires_grad_(True)
    restorer.requires_grad_(False)
    before = {n: p.detach().clone() for n, p in model.state_dict().items()}
    restore_calls = []

    def restore(actual_restorer, image, seed):
        restore_calls.append(seed)
        return actual_restorer(image)

    def loss(logits, batch, cfg, pseudo):
        return (logits - batch['target']).square().mean(), dict(positive_edges=1, negative_edges=1,
                precision=.5, recall=.5, specificity=.5)

    config = dict(backend_adaptation={'pseudo_weight': .5, 'grad_clip': 1.},
                  direct_semantic_affinity={'affinity_loss': {}})
    raw = dict(image=torch.ones(1, 3, 4, 4), target=torch.ones(1, 1, 4, 4))
    optimizer = torch.optim.AdamW(model.affinity_decoder.parameters(), lr=2e-5)
    with patch('models.backend_adaptation.restore_first', restore), \
            patch('train_direct_semantic_affinity.compute_affinity_loss', loss), \
            patch('train_direct_semantic_affinity.move_batch', lambda batch, device: batch), \
            patch.object(optimizer, 'step', wraps=optimizer.step) as step:
        report = current._r1_update(model, restorer, raw, raw, config, 23, 'cpu', optimizer)
    assert step.call_count == 1
    assert restore_calls == [1000023, 2000023]
    assert report['grad_norm'] > 0 and report['auxiliary_weight'] == 0.
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name]) == (not name.startswith('affinity_decoder.'))
    assert all(p.grad is None for p in model.encoder.parameters())
    assert all(p.grad is None for p in restorer.parameters())


def test_export_keeps_exact_fp32_state_and_binds_completed_r1(context):
    from tools.build_semifinal_bundle import state_digest
    r1, d5a = Path(context['artifacts']['r1']), Path(context['artifacts']['d5a'])
    r1.parent.mkdir()
    backend_state = {'weights': torch.arange(4, dtype=torch.float32)}
    restorer_state = {'weights': torch.arange(3, dtype=torch.float32)}
    torch.save(dict(format='phase_affinity_fused_v1', architecture={
        'sam2': {'config_file': 'configs/sam2/x.yaml', 'sam2_repo_path': '/old/root',
                 'input_normalization': 'legacy_none'}}, epoch=20, model_state_dict=backend_state,
        optimizer={'discard': True}), r1)
    torch.save(dict(format='rgb_restoration_diffusion_v1', model_config={
        'architecture': 'pixel_diffusion_nafnet'}, model=restorer_state, optimizer={'discard': True}), d5a)
    status = dict(format='current_r1_fresh_training_v1', status='complete', strict_reload_equal=True,
        updates=1280, final_checkpoint_sha256=current._sha(r1), backend_parameters=4, restorer_parameters=3,
        input_files={str(d5a): current._sha(d5a)}, final_state_sha256=state_digest(backend_state),
        restorer_state_sha256=state_digest(restorer_state))
    (r1.parent / 'status.json').write_text(json.dumps(status), encoding='utf-8')
    receipt = current.run_stage('export_backend', context)
    exported = torch.load(context['artifacts']['backend'], weights_only=True)
    assert torch.equal(exported['model_state_dict']['weights'], backend_state['weights'])
    assert 'optimizer' not in exported
    assert receipt['models']['backend']['fp32_exact']
    assert receipt['models']['restorer']['parameters'] == 3
    assert Path(context['run_dir'], 'export_backend', 'export_manifest.json').is_file()
    with pytest.raises(ValueError, match='already exists'):
        current.run_stage('export_backend', context)


def test_export_rejects_stale_r1_completion_receipt(context):
    r1 = Path(context['artifacts']['r1'])
    r1.parent.mkdir()
    r1.write_bytes(b'not-read')
    (r1.parent / 'status.json').write_text(json.dumps(dict(format='current_r1_fresh_training_v1',
        status='complete', strict_reload_equal=True, updates=1280, final_checkpoint_sha256='0' * 64)),
        encoding='utf-8')
    with pytest.raises(ValueError, match='strictly reloaded'):
        current.run_stage('export_backend', context)


def test_export_rejects_a_different_d5a_than_r1_used(context):
    r1 = Path(context['artifacts']['r1'])
    r1.parent.mkdir()
    r1.write_bytes(b'not-read')
    (r1.parent / 'status.json').write_text(json.dumps(dict(format='current_r1_fresh_training_v1',
        status='complete', strict_reload_equal=True, updates=1280, final_checkpoint_sha256=current._sha(r1),
        input_files={context['artifacts']['d5a']: '0' * 64})), encoding='utf-8')
    with pytest.raises(ValueError, match='different from the frozen restorer'):
        current.run_stage('export_backend', context)


def test_generated_configs_never_overwrite_existing_private_recipe(context):
    with pytest.raises(ValueError, match='already exists'):
        current._save_config(context, 'd5a', {})


def test_stage_output_cannot_escape_run_directory(context, tmp_path):
    context['artifacts']['r1'] = str(tmp_path / 'unrelated' / 'final.pt')
    with pytest.raises(ValueError, match='inside run_dir'):
        current.prepare_r1_config(context)


def test_source_identity_binds_pseudo_image_and_mask_without_loading_arrays(context):
    raw = Path(context['assets']['raw_dir'], 'train_000.jpg')
    completed = Path(context['assets']['completed_gt_dir'])
    (completed / 'train_000_gt.npz').write_bytes(b'not-loaded')
    (completed / 'train_000_class.json').write_text('{}', encoding='utf-8')
    image = Path(context['assets']['sam2_source_dir'], 'source.jpg')
    image.write_bytes(b'authorized-training-source')
    mask = Path(context['assets']['sam2_geometry_dir'], 'mask.npz')
    mask.write_bytes(b'not-loaded-pseudo-target')
    canonical = SimpleNamespace(samples=[raw], completed_gt_dir=completed)
    pseudo = SimpleNamespace(rows=[{'source_sha256': current._sha(image), 'mask_file': mask.name}],
                             _source_path=lambda row: image)
    wrap = lambda original: SimpleNamespace(whole=SimpleNamespace(base_dataset=SimpleNamespace(base=original)))
    receipt = current._source_files(context, wrap(canonical), wrap(pseudo))
    assert receipt[str(image)] == current._sha(image)
    assert receipt[str(mask)] == current._sha(mask)
    image.write_bytes(b'changed-source')
    with pytest.raises(ValueError, match='source bytes differ'):
        current._source_files(context, wrap(canonical), wrap(pseudo))
