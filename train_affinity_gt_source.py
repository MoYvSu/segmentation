# -*- coding: utf-8 -*-
"""独立五轮同源GT对照：仅几何ID不同，共同覆盖、输入与原64个SAM2样本固定。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import time
from unittest.mock import patch

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.affinity_native import choose_box, geometry_sample, NativeDegradationDataset, NativeViewDataset
from data.affinity_mixed import WholeViewDataset, view_plan
from data.rgb_restoration_dataset import read_rgb
from models.backend_adaptation import load_backend, restore_first, save_backend
from tools.probe_affinity_structure import decode_reflect
from tools.run_marker_reflect import save_prediction, precision_record, config_gate, source_snapshot
from tools.run_affinity_sweep import runtime_contract
from train_affinity_connectivity import configure_training, frozen_digest, get_restorer, draw_receipt
from train_affinity_coverage import load_original_core
from train_backend_adaptation import build_datasets as original_datasets, sha, tensor_digest, write_json
from train_direct_semantic_affinity import compute_affinity_loss, move_batch, set_seed
from utils.affinity_loss import build_affinity_targets_torch
from utils.config import load_config, project_path

ROOT = Path(__file__).resolve().parent
FORMAT = 'affinity_gt_source_v1'
HEAD_FORMAT = 'affinity_gt_source_head_v1'
R1_SHA = '43da103c95c7f85149bd0533fd116d26f0a81c91d793db7e8938b7d4ae80e434'
D5A_SHA = '048954795dc634f82fecf96d5e02bb2f2505afe5eb76298dff5a1b0713b3c80a'
COMMON_KEYS = ('epoch', 'step', 'seed', 'manual', 'pseudo', 'manual_draw', 'pseudo_draw',
               'manual_crop', 'pseudo_crop', 'training_view', 'inputs')


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode('utf-8')).hexdigest()


def tensor_sha(value):
    return tensor_digest([('value', value)])


def relative_input(config, value):
    """新manifest只接受项目内的相对输入；不借绝对路径绕过来源白名单。"""
    base = Path(config['paths']['project_root']).resolve()
    p = Path(value)
    if p.is_absolute() or '..' in p.parts or '\\' in str(value) or ':' in str(value):
        raise ValueError('Manifest inputs must use project-relative paths')
    result = (base / p).resolve()
    if not result.is_relative_to(base) or not result.is_file():
        raise ValueError('Manifest input missing or outside project')
    return result


def validate_pair(arrays, manual_full, shape):
    """来源覆盖只审计；绝不把teacher标签标作人工原覆盖真值。"""
    required = {'manual_ids', 'teacher_ids', 'common_known', 'original_covered', 'filled'}
    if not required.issubset(arrays):
        raise ValueError('Paired target fields incomplete')
    a, b = np.asarray(arrays['manual_ids']), np.asarray(arrays['teacher_ids'])
    if any(x.shape != tuple(shape) or x.dtype.kind not in 'iu' or
           np.any(x < 0) or np.any(x > 65535) for x in (a, b)):
        raise ValueError('Native integer instance targets must satisfy uint16 range')
    masks = {}
    for name in ('common_known', 'original_covered', 'filled'):
        x = np.asarray(arrays[name])
        if x.shape != tuple(shape) or x.dtype.kind not in 'bu' or not np.isin(x, [0, 1]).all():
            raise ValueError('Native provenance masks must be binary and matching shape')
        masks[name] = x.astype(bool)
    known = masks['common_known']
    if (not known.any() or not np.array_equal(known, (a > 0) & (b > 0))
            or np.any(a[~known]) or np.any(b[~known])):
        raise ValueError('Targets must share the exact common-known domain; outside remains zero')
    if manual_full.shape != a.shape or not np.array_equal(a, np.where(known, manual_full, 0)):
        raise ValueError('Manual labels differ from original processed GT inside common domain')
    if (masks['original_covered'] & masks['filled']).any():
        raise ValueError('Original and filled provenance overlap')
    return dict(manual_ids=a, teacher_ids=b, **masks)


def load_targets(config, arm, *, expected_sources=32):
    if arm not in ('manual', 'teacher'):
        raise ValueError('Unknown arm')
    path = relative_input(config, config['gt_source']['manifest'])
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if (manifest.get('format') != 'affinity_gt_source_targets_v1'
            or manifest.get('status') != 'complete'
            or manifest.get('teacher_checkpoint_sha256') != R1_SHA
            or manifest.get('restorer_checkpoint_sha256') != D5A_SHA
            or manifest.get('semantic_labels_used_for_teacher_training') is not False):
        raise ValueError('Completed class-agnostic R1/D5a manifest required')
    rows = manifest.get('rows', [])
    if len(rows) != expected_sources or len({r['source'] for r in rows}) != len(rows):
        raise ValueError('Exact unique source cohort required')
    if len({r['image_sha256'] for r in rows}) != len(rows):
        raise ValueError('Duplicate source image SHA')
    protected = {str(path): sha(path)}
    if arm == 'teacher':
        try:
            approval_path = relative_input(config, config['gt_source']['approval_file'])
            approval = json.loads(approval_path.read_text(encoding='utf-8'))
        except (KeyError, ValueError, FileNotFoundError) as error:
            raise RuntimeError('Teacher candidates require explicit_user_confirmed quality review') from error
        if (approval.get('status') != 'explicit_user_confirmed'
                or approval.get('manifest_sha256') != protected[str(path)]
                or approval.get('review_checked_target_count') != expected_sources):
            raise RuntimeError('Teacher candidates require explicit_user_confirmed review bound to this manifest')
        protected[str(approval_path)] = sha(approval_path)
    for row in rows:
        for field, digest in [('image', 'image_sha256'), ('manual_gt', 'manual_gt_sha256'),
                              ('pair_npz', 'pair_sha256')]:
            p = relative_input(config, row[field])
            if sha(p) != row[digest]:
                raise RuntimeError('Manifest input SHA differs: ' + field)
            protected[str(p)] = row[digest]
        image_path = relative_input(config, row['image'])
        if image_path.stem != row['source'] or not row['source'].startswith('train_'):
            raise ValueError('Source identity must be an allowed training stem')
        shape = read_rgb(str(image_path)).shape[:2]
        with np.load(relative_input(config, row['manual_gt']), allow_pickle=False) as gt:
            full = gt['instance_map']
            with np.load(relative_input(config, row['pair_npz']), allow_pickle=False) as pair:
                checked = validate_pair(pair, full, shape)
                for key in ('original_covered', 'filled'):
                    if key not in gt or not np.array_equal(checked[key], gt[key].astype(bool)):
                        raise ValueError('Paired provenance differs from processed manual source')
    return manifest, protected


def load_training_targets(config, arm):
    """原候选确认保持原门禁；零缝派生监督另绑定独立manifest。"""
    manifest, protected = load_targets(config, arm)
    if config['gt_source'].get('seam_repair') is None:
        return manifest, protected
    if arm != 'teacher':
        raise ValueError('Zero-seam support is an explicit teacher-only variant')
    from data.teacher_seam_manifest import load_seam_manifest
    # 实际变体依赖必须在source_snapshot之前导入，才进入严格续训源码封存。
    import utils.teacher_seam_training
    parent_sha = sha(relative_input(config, config['gt_source']['manifest']))
    derived, extra = load_seam_manifest(config, manifest, parent_sha)
    repaired = {r['source']: r for r in derived['rows']}
    manifest = deepcopy(manifest)
    for row in manifest['rows']:
        row['seam_pair_npz'] = repaired[row['source']]['pair_npz']
        row['seam_pair_sha256'] = repaired[row['source']]['pair_sha256']
    protected.update(extra)
    return manifest, protected


class PairGeometrySource(Dataset):
    """同一清晰RGB和common mask决定裁窗，只有输出实例ID取决于arm。"""
    def __init__(self, config, rows, arm, *, image_size=1024, grid=512, seed=42):
        self.config, self.rows, self.arm = config, rows, arm
        self.image_size, self.grid, self.seed = image_size, grid, seed
        self.epoch, self.draw, self.size = 0, 0, 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[int(index)]
        path = relative_input(self.config, row['image'])
        rgb = read_rgb(str(path))
        with np.load(relative_input(self.config, row['pair_npz']), allow_pickle=False) as blob:
            ids = blob[self.arm + '_ids'].copy()
            common = blob['common_known'].astype(bool)
        if 'seam_pair_npz' in row:
            if self.arm != 'teacher':
                raise ValueError('Repaired IDs must not alter the manual control')
            with np.load(relative_input(self.config, row['seam_pair_npz']), allow_pickle=False) as blob:
                ids = blob['teacher_ids'].copy()
        shape = ids.shape
        box, attempt = (0, 0, *shape), 0
        if self.size:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, self.draw, 90]))
            opts = self.config['affinity_native']
            box, attempt = choose_box(common.astype(np.uint8), self.size, rng,
                attempts=opts['crop_attempts'], minimum_pairs=opts['minimum_known_pairs'])
        y, x, y1, x1 = box
        sample = geometry_sample(rgb[y:y1, x:x1], ids[y:y1, x:x1],
                                 image_size=self.image_size, grid=self.grid)
        # 二者采用同一nearest算子，标签0和padding同时保持ignore。
        known = sample['affinity_instance_map'] > 0
        sample['valid_content'] &= known[None]
        sample['affinity_valid_content'] = sample['valid_content']
        sample.update(name=path.name, image_name=path.name, image_path=str(path), source_kind='manual',
            crop_box=torch.tensor(box, dtype=torch.int32), crop_attempt=attempt,
            native_source_shape=torch.tensor(shape, dtype=torch.int32), native_crop_size=self.size)
        return sample


def epoch_order(seed, epoch, stream, count):
    generator = torch.Generator().manual_seed(seed + epoch * 2 + stream)
    return torch.randperm(count, generator=generator).tolist()


class IndependentMixedDataset(Dataset):
    """独立均匀50/50表，不读取旧control训练回执。"""
    def __init__(self, original, kind, config, *, pair_rows=None, arm=None):
        self.seed = config['gt_source']['seed']
        self.stream = 0 if kind == 'manual' else 1
        self.mapping = None
        base = (PairGeometrySource(config, pair_rows, arm, seed=original.seed,
                                  image_size=original.base_dataset.image_size,
                                  grid=original.base_dataset.affinity_grid) if kind == 'manual'
                else None)
        self.pair_base = base
        whole = base if base is not None else WholeViewDataset(original.base_dataset, 'pseudo')
        native = base if base is not None else NativeViewDataset(original.base_dataset, 'pseudo',
                    1024, original.seed, config['affinity_native'])
        args = (original.degradation, original.noise, original.seed, original.repeats, original.augment)
        self.whole = NativeDegradationDataset(whole, *args)
        self.native = NativeDegradationDataset(native, *args)

    def __len__(self):
        return len(self.whole)

    def set_epoch(self, epoch):
        order = epoch_order(self.seed, epoch, self.stream, len(self))
        plan = view_plan(self.seed, epoch, len(self))
        self.mapping = dict(zip(order, plan))
        self.whole.set_epoch(epoch)
        self.native.set_epoch(epoch)

    def __getitem__(self, index):
        if self.mapping is None:
            raise RuntimeError('Set epoch first')
        size = self.mapping[int(index)]
        if self.pair_base is not None:
            self.pair_base.size = size
        item = (self.native if size else self.whole)[index]
        item['training_view'] = 'native1024' if size else 'whole'
        return item


def verify_canonical_rows(base, rows, config):
    """RGB和人工GT都绑定既有builder；不接受形状碰巧相同的另一个GT。"""
    canonical_sources = {p.stem: p.resolve() for p in base.samples}
    if set(canonical_sources) != {r['source'] for r in rows}:
        raise ValueError('Teacher rows differ from all32 canonical manual sources')
    for row in rows:
        if canonical_sources[row['source']] != relative_input(config, row['image']):
            raise ValueError('Manifest source is not the canonical allowed manual image')
        gt = (base.completed_gt_dir/(row['source']+'_gt.npz')).resolve()
        if gt != relative_input(config, row['manual_gt']):
            raise ValueError('Manual GT is not the canonical processed source')


def build_datasets(config, arm, manifest):
    # 原builder保留32人工身份、既有64 SAM2授权和退化配方，绝不读取旧20轮控制输出。
    manual, pseudo, metadata = original_datasets(config)
    verify_canonical_rows(manual.base_dataset, manifest['rows'], config)
    if len(manual) != 64 or len(pseudo) != 64 or metadata['sam2_sources'] != 64:
        raise ValueError('Original32x2 +64SAM2 budget required')
    rows = sorted(manifest['rows'], key=lambda r: r['source'])
    protected = {}
    for p in (manual.base_dataset.completed_gt_dir/'manifest.json',
              pseudo.base_dataset.dataset_dir/'manifest.jsonl',
              Path(project_path(config, config['backend_adaptation']['sam2_geometry']['reuse_authorization_doc']))):
        protected[str(p)] = sha(p)
    for path in manual.base_dataset.samples:
        for p in (path.with_suffix('.json'), manual.base_dataset.completed_gt_dir/(path.stem+'_class.json')):
            protected[str(p)] = sha(p)
    for row in pseudo.base_dataset.rows:
        for p in (pseudo.base_dataset._source_path(row), pseudo.base_dataset.dataset_dir/row['mask_file']):
            protected[str(p)] = sha(p)
    metadata['protected_inputs'] = protected
    return (IndependentMixedDataset(manual, 'manual', config, pair_rows=rows, arm=arm),
            IndependentMixedDataset(pseudo, 'pseudo', config), metadata)


def batch_signature(batch, restored):
    ids = batch.get('affinity_instance_map', batch['instance_map'])
    pixels = batch.get('affinity_valid_content', batch['valid_content'])
    target, valid = build_affinity_targets_torch(ids, pixels)
    return dict(image=tensor_sha(batch['image']), clean_image=tensor_sha(batch['clean_image']),
                restored=tensor_sha(restored), valid_pixels=tensor_sha(pixels),
                valid_edges=tensor_sha(valid), shape=list(target.shape))


def compare_pair_rows(row, reference):
    for key in COMMON_KEYS:
        if row[key] != reference[key]:
            raise RuntimeError('A/B common draw or input differs: ' + key)


@torch.no_grad()
def initial_logits_parity(head, reference, features):
    modes = [(m, m.training) for m in head.modules()]
    head.eval(); reference.eval()
    try:
        actual = head(features)['affinity_logits']
        expected = reference(features)['affinity_logits']
        if not torch.equal(actual, expected):
            raise RuntimeError('Actual initial frozen-feature logits differ after strict copy')
        return dict(passed=True, logits_sha256=tensor_sha(actual), shape=list(actual.shape))
    finally:
        for module, mode in modes:
            module.training = mode


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def schedule_factor(epoch, schedule_epochs=20, floor=.1):
    return floor + (1-floor) * (1+math.cos(math.pi*epoch/schedule_epochs))/2


def validate_config(config):
    opts = config['gt_source']
    if (opts['format'] != FORMAT or opts['epochs'] != 5 or opts['schedule_epochs'] != 20
            or opts['updates_per_epoch'] != 64 or opts['snapshots'] != [0, 1, 5]
            or opts['checkpoint_sha256'] != R1_SHA or opts['learning_rate'] != 2e-5
            or config['backend_adaptation']['num_workers'] != 0
            or config['backend_adaptation']['restoration']['checkpoint_sha256'] != D5A_SHA
            or config['direct_semantic_affinity']['affinity_loss']['manual_uncovered_as_boundary']):
        raise ValueError('Fixed five-epoch GT-source recipe changed')
    if config['inference'].get('marker_partition_restore', {}).get('enabled', False):
        raise ValueError('Original marker partition restore must remain disabled')


def checkpoint_payload(model, optimizer, scheduler, config, arm, epoch, updates, contract):
    return dict(format=HEAD_FORMAT, version=1, arm=arm, epoch=epoch, updates=updates,
        geometry_state_dict={k: v.detach().cpu().clone() for k, v in model.affinity_decoder.state_dict().items()},
        optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), rng=rng_state(),
        config=deepcopy(config), contract=deepcopy(contract),
        head_state_sha256=tensor_digest(model.affinity_decoder.state_dict().items()))


def strict_resume(payload, head, optimizer, scheduler, contract, arm):
    if (payload.get('format') != HEAD_FORMAT or payload.get('version') != 1
            or payload.get('arm') != arm or payload.get('contract') != contract
            or payload['updates'] != payload['epoch'] * 64 or not 1 <= payload['epoch'] <= 20):
        raise ValueError('Same-arm versioned epoch-boundary resume required')
    head.load_state_dict(payload['geometry_state_dict'], strict=True)
    if tensor_digest(head.state_dict().items()) != payload['head_state_sha256']:
        raise RuntimeError('Strict restored head state differs')
    optimizer.load_state_dict(payload['optimizer']); scheduler.load_state_dict(payload['scheduler'])
    restore_rng(payload['rng'])
    return payload['epoch'], payload['updates']


def verify_checkpoint_log(payload, path):
    path = Path(path)
    rows = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
    if (sha(path) != payload.get('draw_log_sha256') or len(rows) != payload['updates']
            or not rows or rows[-1]['updates'] != payload['updates']):
        raise ValueError('Resume checkpoint and exact committed draw log differ')
    return rows


def validate_peer_status(status, contract, arm, end, *, allow_prefix=False):
    budget = status.get('updates', 0)
    if (status.get('format') != FORMAT or status.get('status') != 'complete'
            or status.get('arm') == arm or status.get('arm') not in ('manual', 'teacher')
            or status.get('contract') != contract or not isinstance(budget, int)
            or budget < 64 or budget % 64 or budget > end*64
            or (not allow_prefix and budget != end*64)
            or not all(status.get(k) is True for k in ('frozen_unchanged', 'restorer_unchanged', 'strict_reload_equal'))):
        raise ValueError('Complete same-contract opposite arm required for paired comparison')


def append(path, row):
    with Path(path).open('a', encoding='utf-8') as out:
        out.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


@torch.no_grad()
def save_monitor(model, restorer, config, output, epoch, rows):
    """四张固定全图缩略图；使用sealed真实视野和reflect种子，无指标选轮。"""
    from data.mim_dataset import list_images
    views = load_original_core(config).coverage_original_views
    folder = Path(output)/'monitor'/f'e{epoch:02d}'; folder.mkdir(parents=True, exist_ok=True)
    opts = config['gt_source']
    references = {}
    if epoch == 0:
        reference_path = Path(project_path(config, opts['monitor_reference_report']))
        if sha(reference_path) != opts['monitor_reference_sha256']:
            raise RuntimeError('Sealed reflect100 monitor reference report differs')
        reference = json.loads(reference_path.read_text(encoding='utf-8'))
        if (not reference.get('full100_complete') or reference['checkpoint_sha256'] != R1_SHA
                or reference['restorer_checkpoint_sha256'] != D5A_SHA):
            raise RuntimeError('Complete existing best deployment receipt required')
        references = {r['source']: r for r in reference['images']}
    tests = [Path(p) for p in list_images(project_path(config, config['inference']['test_dir']))]
    train_rows = sorted(rows, key=lambda r: r['source'])
    chosen = [(p, config['backend_adaptation']['restoration']['inference_seed'] + i, 'test')
              for i, p in enumerate(tests) if p.stem in opts['monitor_images']]
    chosen += [(relative_input(config, row['image']),
                int(row.get('seed', config['backend_adaptation']['restoration']['inference_seed'] + i)), 'training')
               for i, row in enumerate(train_rows) if row['source'] in opts['train_monitors']]
    if len(chosen) != len(opts['monitor_images'])+len(opts['train_monitors']):
        raise ValueError('Fixed monitor samples missing')
    before = tensor_digest(model.state_dict().items())

    def reflected(semantic, boundary, actual_config):
        result, _, _ = decode_reflect(semantic, boundary[0, 0].numpy(), actual_config)
        return result['instances'], result['classes']

    record = []
    with patch.object(views, 'decode_native', side_effect=reflected):
        for path, seed, kind in chosen:
            rgb, predictions = views.predict_views(model, restorer, path, config, 'cuda', seed, [1024])
            value = predictions['patch1024']; ids = value['instances']; classes = value['classes']
            final_hashes = save_prediction(folder, path.stem, ids, classes)
            parity = None
            if epoch == 0:
                if kind == 'test':
                    old = references[path.stem]
                    expected = old['output']
                    if seed != old['seed'] or sha(path) != old['source_sha256']:
                        raise RuntimeError('Existing test monitor source/seed differs')
                else:
                    expected = next(r['prediction_hashes'] for r in rows if r['source'] == path.stem)
                if final_hashes != expected:
                    raise RuntimeError('Actual e0 final PNG/classes differs from existing R1 reflect baseline')
                parity = True
            h, w = rgb.shape[:2]; width = 300; height = max(1, round(h*width/w))
            phase = np.full((*ids.shape, 3), 70, np.uint8)
            for label, cls in classes.items():
                phase[ids == int(label)] = (40, 190, 230) if cls == 1 else (220, 100, 50)
            contours = np.zeros(ids.shape, bool)
            contours[1:] |= ids[1:] != ids[:-1]; contours[:, 1:] |= ids[:, 1:] != ids[:, :-1]
            phase[contours] = 255
            heat = cv2.applyColorMap(np.rint(np.clip(value['boundary'], 0, 1)*255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            panels = [cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), heat, phase]
            board = np.concatenate([cv2.resize(p, (width, height), interpolation=cv2.INTER_AREA) for p in panels], axis=1)
            if not cv2.imwrite(str(folder/(path.stem+'.png')), board):
                raise RuntimeError('Monitor PNG could not be saved')
            record.append(dict(source=path.stem, kind=kind, seed=seed, GT_used_for_display=False,
                               final_output_hashes=final_hashes, existing_R1_e0_exact=parity,
                               boundary_sha256=hashlib.sha256(value['boundary'].tobytes()).hexdigest()))
    if tensor_digest(model.state_dict().items()) != before:
        raise RuntimeError('Monitor changed frozen or training state')
    write_json(folder/'receipt.json', dict(epoch=epoch, rows=record,
               scope='process observation only; test GT absent; no checkpoint selection'))


def train(config, arm, output, *, resume=None, until_epoch=None, paired_steps=None, config_path=None):
    validate_config(config)
    opts, cfg = config['gt_source'], config['backend_adaptation']
    manifest, protected = load_training_targets(config, arm)
    repair = opts.get('seam_repair')
    repair_rows = {r['source']: r for r in manifest['rows']} if repair else None
    output = Path(project_path(config, output)).resolve()
    root = Path(config['paths']['project_root']).resolve()
    if not output.is_relative_to(root/'outputs'):
        raise ValueError('Training artifacts must remain in ignored project outputs')
    if not torch.cuda.is_available():
        raise RuntimeError('Actual sam2_env CUDA required; CPU tests are not training')
    end = opts['epochs'] if until_epoch is None else int(until_epoch)
    if not 1 <= end <= opts['schedule_epochs']:
        raise ValueError('Epoch stop must remain within the existing20-epoch schedule')
    if until_epoch is not None and resume is None and end != opts['epochs']:
        raise ValueError('Longer work requires explicit same-run resume')
    if repair and end != 5:
        raise ValueError('This repaired/control comparison is sealed to five epochs')
    if output.exists() and resume is None:
        raise FileExistsError('Fresh output required; existing work is never overwritten')
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); cv2.setNumThreads(2); set_seed(opts['seed'])
    base_config = deepcopy(config); base_config.pop('gt_source')
    config_gate(base_config, manifest['config'])
    if precision_record() != manifest['precision'] or runtime_contract(config) != manifest['runtime']:
        raise RuntimeError('Teacher and actual training runtime/numerical settings differ')
    for relative, digest in manifest['protected_code'].items():
        p = Path(project_path(config, relative))
        if sha(p) != digest:
            raise RuntimeError('Sealed teacher inference code changed: '+relative)
        protected[str(p)] = digest
    reference_report = Path(project_path(config, opts['monitor_reference_report']))
    if sha(reference_report) != opts['monitor_reference_sha256']:
        raise RuntimeError('Best deployment baseline report differs')
    protected[str(reference_report)] = opts['monitor_reference_sha256']
    checkpoint = Path(project_path(config, opts['checkpoint']))
    if sha(checkpoint) != R1_SHA:
        raise RuntimeError('Strict initial R1 checkpoint changed')
    protected[str(checkpoint)] = R1_SHA
    restoration_path = Path(project_path(config, cfg['restoration']['checkpoint']))
    protected[str(restoration_path)] = D5A_SHA
    protected[str(Path(__file__).resolve())] = sha(__file__)
    if config_path:
        protected[str(Path(config_path).resolve())] = sha(config_path)
    model, bundle = load_backend(checkpoint, config, 'cuda')
    restorer = get_restorer(config, 'cuda')
    total = sum(p.numel() for p in model.parameters())+sum(p.numel() for p in restorer.parameters())
    if total >= 500_000_000 or bundle['epoch'] != 20:
        raise ValueError('R1 epoch or parameter budget differs')
    manual, pseudo, data = build_datasets(config, arm, manifest)
    protected.update(data.pop('protected_inputs'))
    # 额外封存训练实际导入的源码；教师的推理清单未必含全部训练特有模块。
    if resume:
        old_receipt = json.loads((output/'status.json').read_text(encoding='utf-8'))
        training_code = old_receipt['training_code']
        training_virtual = old_receipt['training_virtual_code_paths']
        for relative, digest in training_code.items():
            sealed = output/'training_code'/'source'/relative
            if sha(sealed) != digest or sha(project_path(config, relative)) != digest:
                raise RuntimeError('Exact saved training source changed before resume: '+relative)
    else:
        training_code, training_virtual = source_snapshot(output/'training_code')
    for relative, digest in training_code.items():
        protected[str(Path(project_path(config, relative)))] = digest
    head = model.affinity_decoder
    frozen = frozen_digest(model); restored_state = tensor_digest(restorer.state_dict().items())
    initial_head = tensor_digest(head.state_dict().items())
    reference = deepcopy(head).eval().requires_grad_(False)
    reference.load_state_dict(head.state_dict(), strict=True)
    if tensor_digest(reference.state_dict().items()) != initial_head:
        raise RuntimeError('Initial strict head copy differs')
    contract = dict(format=FORMAT, recipe_sha256=canonical_sha(config), manifest_sha256=sha(relative_input(config, opts['manifest'])),
                    checkpoint_sha256=R1_SHA, restorer_checkpoint_sha256=D5A_SHA, frozen_state_sha256=frozen,
                    restorer_state_sha256=restored_state, initial_head_sha256=initial_head,
                    training_code_sha256=canonical_sha(training_code))
    if repair:
        contract.update(variant='teacher_zero_seam_v1', seam_manifest_sha256=repair['manifest_sha256'],
                        baseline_steps_sha256=repair['baseline_steps_sha256'],
                        baseline_status_sha256=repair['baseline_status_sha256'])
    configure_training(model)
    optimizer = torch.optim.AdamW(head.parameters(), lr=opts['learning_rate'], weight_decay=cfg['weight_decay'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda e: schedule_factor(e, opts['schedule_epochs'], cfg['minimum_lr_ratio']))
    start, updates = 0, 0
    previous_rows, previous_status = [], {}
    if resume:
        payload = torch.load(project_path(config, resume), map_location='cpu', weights_only=False)
        start, updates = strict_resume(payload, head, optimizer, scheduler, contract, arm)
        previous_rows = verify_checkpoint_log(payload, output/'steps.jsonl')
        previous_status = json.loads((output/'status.json').read_text(encoding='utf-8'))
        if previous_status.get('contract') != contract or previous_status.get('arm') != arm:
            raise ValueError('Resume receipt differs from the same-arm source contract')
        if start >= end:
            raise ValueError('Resume stop must be later than saved epoch')
    elif any((output/p).exists() for p in ('steps.jsonl', 'last_head.pt')):
        raise FileExistsError('Existing logs or weights cannot be replaced')
    if repair:
        configured_peer = relative_input(config, repair['baseline_steps'])
        if paired_steps and Path(paired_steps).resolve() != configured_peer:
            raise ValueError('Repair comparison must use the sealed existing teacher draw log')
        other = configured_peer
    else:
        other = Path(paired_steps) if paired_steps else output.parent/('teacher' if arm == 'manual' else 'manual')/'steps.jsonl'
    paired = [json.loads(s) for s in other.read_text(encoding='utf-8').splitlines()] if other.is_file() else []
    if paired:
        peer_status = other.parent/'status.json'
        peer_receipt = json.loads(peer_status.read_text(encoding='utf-8'))
        if repair:
            from utils.teacher_seam_training import validate_repaired_peer, compare_repaired_rows
            if (peer_status.resolve() != relative_input(config, repair['baseline_status'])
                    or sha(other) != repair['baseline_steps_sha256']
                    or sha(peer_status) != repair['baseline_status_sha256']):
                raise RuntimeError('Sealed original teacher control files changed')
            peer_check = validate_repaired_peer(peer_receipt, contract, config, end)
            for relative, digest in peer_receipt['training_code'].items():
                sealed_peer = other.parent/'training_code'/'source'/relative
                if sha(sealed_peer) != digest:
                    raise RuntimeError('Original teacher sealed training snapshot changed: '+relative)
                if relative != 'train_affinity_gt_source.py' and sha(project_path(config, relative)) != digest:
                    raise RuntimeError('Original teacher training dependency changed: '+relative)
        else:
            validate_peer_status(peer_receipt, contract, arm, end, allow_prefix=bool(resume))
        if len(paired) != peer_receipt['updates'] or sha(other) != peer_receipt.get('draw_log_sha256'):
            raise ValueError('Other arm has an incomplete comparison draw budget')
        for row, peer in zip(previous_rows, paired):
            (compare_repaired_rows if repair else compare_pair_rows)(row, peer)
        protected[str(other.resolve())] = sha(other)
        protected[str(peer_status.resolve())] = sha(peer_status)
    elif repair:
        raise ValueError('Original complete teacher draw budget is mandatory for seam repair')
    status = dict(format=FORMAT, arm=arm, status='running', config=config, contract=contract,
        parameter_count=total, initialization='strict complete R1 e20; Stage1/joint-v3 spectrum inherited',
        protected_inputs=protected, data=data, optimizer=dict(name='AdamW', eps=1e-4),
        epochs_requested=end, schedule_epochs=20, updates_per_epoch=64, semantic_supervision=False,
        train_modes='affinity_decoder.train; remaining model.eval; actual decoder GroupNorm, no BN/dropout',
        paired_reference=str(other) if paired else None, official_score=None,
        training_code=training_code, training_virtual_code_paths=training_virtual)
    if repair:
        status.update(variant=contract['variant'], reused_teacher_control=peer_check,
                      changed_existing_training_code=['train_affinity_gt_source.py'],
                      supervision_change='Only known teacher-zero seam assignment; old legal targets unchanged; per-channel normalization recomputed')
    if previous_status.get('actual_initial_parity'):
        status['actual_initial_parity'] = previous_status['actual_initial_parity']
    if previous_status.get('initial_peer_sam_loss_equal') is True:
        status['initial_peer_sam_loss_equal'] = True
    write_json(output/'status.json', status)
    if not start:
        save_monitor(model, restorer, config, output, 0, manifest['rows'])
    started = time.time()
    initial_parity = None
    for epoch in range(start+1, end+1):
        manual.set_epoch(epoch); pseudo.set_epoch(epoch)
        loaders = [DataLoader(ds, batch_size=1, sampler=epoch_order(opts['seed'], epoch, i, len(ds)),
                   num_workers=0, pin_memory=True) for i, ds in enumerate((manual, pseudo))]
        configure_training(model)
        count = 0
        for step, (raw, raw_sam) in enumerate(zip(*loaders), start=1):
            seed = opts['seed'] + epoch*10000 + step-1; set_seed(seed)
            if raw['training_view'] != raw_sam['training_view']:
                raise RuntimeError('Two source streams must share view within update')
            row = dict(epoch=epoch, step=step, seed=seed, manual=raw['name'], pseudo=raw_sam['name'],
                manual_draw=draw_receipt(raw), pseudo_draw=draw_receipt(raw_sam),
                manual_crop=raw['crop_box'].tolist(), pseudo_crop=raw_sam['crop_box'].tolist(),
                training_view=raw['training_view'][0], inputs=[])
            optimizer.zero_grad(set_to_none=True); losses=[]; supervision=[]
            for is_sam, raw_batch in [(False, raw), (True, raw_sam)]:
                batch = move_batch(raw_batch, 'cuda')
                with torch.no_grad():
                    restored = restore_first(restorer, batch['image'], seed+(2000000 if is_sam else 1000000))
                    row['inputs'].append(batch_signature(batch, restored))
                    if repair and not is_sam:
                        from utils.teacher_seam_training import audit_repaired_batch
                        row['seam_support'] = audit_repaired_batch(batch, config, repair_rows)
                    features = model.encoder(restored)
                    if not start and updates == 0 and not is_sam:
                        initial_parity = initial_logits_parity(head, reference, features)
                        status['actual_initial_parity'] = initial_parity
                logits = head(features)['affinity_logits']
                loss, details = compute_affinity_loss(logits, batch,
                    config['direct_semantic_affinity']['affinity_loss'], pseudo=is_sam)
                if not torch.isfinite(loss) or details['positive_edges']+details['negative_edges'] <= 0:
                    raise FloatingPointError('Finite supervised geometry loss required')
                (loss*(cfg['pseudo_weight'] if is_sam else 1.)).backward()
                losses.append(float(loss.detach()))
                supervision.append({k: details[k] for k in ('positive_edges','negative_edges','precision','recall','specificity')})
                del loss, features, logits, restored, batch
            if updates < len(paired):
                (compare_repaired_rows if repair else compare_pair_rows)(row, paired[updates])
                if repair and updates == 0:
                    if losses[1] != paired[0]['losses'][1]:
                        raise RuntimeError('Initial same-state SAM2 loss differs from original teacher')
                    status['initial_peer_sam_loss_equal'] = True
            grad = float(torch.nn.utils.clip_grad_norm_(head.parameters(), cfg['grad_clip'], error_if_nonfinite=True))
            if grad <= 0 or any(p.grad is not None for n,p in model.named_parameters() if not n.startswith('affinity_decoder.')):
                raise RuntimeError('Only the affinity head may receive a finite nonzero gradient')
            optimizer.step(); updates += 1; count += 1
            row.update(updates=updates, arm=arm, losses=losses, supervision=supervision, grad_norm=grad)
            append(output/'steps.jsonl', row)
            if step == 1 or step % 16 == 0:
                print('GT_SOURCE_STEP', arm, epoch, step, losses, flush=True)
        if count != 64 or updates != epoch*64:
            raise RuntimeError('Actual64-update epoch incomplete')
        scheduler.step()
        payload = checkpoint_payload(model, optimizer, scheduler, config, arm, epoch, updates, contract)
        payload['draw_log_sha256'] = sha(output/'steps.jsonl')
        torch.save(payload, output/'last_head.pt')
        check = deepcopy(head)
        loaded = torch.load(output/'last_head.pt', map_location='cpu', weights_only=False)
        check.load_state_dict(loaded['geometry_state_dict'], strict=True)
        if tensor_digest(check.state_dict().items()) != loaded['head_state_sha256']:
            raise RuntimeError('Saved epoch strict head reload differs')
        del check, loaded
        if epoch in opts['snapshots']:
            save_monitor(model, restorer, config, output, epoch, manifest['rows'])
        status.update(epoch=epoch, updates=updates, next_lr=optimizer.param_groups[0]['lr'],
                      elapsed_seconds=time.time()-started, draw_log_sha256=payload['draw_log_sha256'])
        write_json(output/'status.json', status)
    if frozen_digest(model) != frozen or tensor_digest(restorer.state_dict().items()) != restored_state:
        raise RuntimeError('Frozen upstream or D5a changed')
    for p, digest in protected.items():
        if sha(p) != digest:
            raise RuntimeError('Protected training input or code changed: '+p)
    final_state = tensor_digest(model.state_dict().items())
    save_backend(model, config, bundle['backend_adaptation'], output/'final.pt', end,
                 extra=dict(format=FORMAT, arm=arm, updates=updates, contract=contract))
    del payload, head, model, optimizer, reference; gc.collect(); torch.cuda.empty_cache()
    reloaded, _ = load_backend(output/'final.pt', config, 'cuda')
    if tensor_digest(reloaded.state_dict().items()) != final_state:
        raise RuntimeError('Strict complete fused model reload differs')
    status.update(status='complete', frozen_unchanged=True, restorer_unchanged=True,
        strict_reload_equal=True, final_checkpoint_sha256=sha(output/'final.pt'),
        final_state_sha256=final_state, checkpoint='final.pt', head_checkpoint='last_head.pt',
        selection='fixed_phase_endpoint_no_validation_or_promotion',
        paired_rows_verified=min(len(paired), updates), paired_actual_rows_verified=len(paired)==updates)
    write_json(output/'status.json', status)
    print('GT_SOURCE_COMPLETE', arm, end, updates, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/train/affinity_gt_source.yaml')
    parser.add_argument('--arm', choices=['manual','teacher'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume'); parser.add_argument('--until-epoch', type=int)
    parser.add_argument('--paired-steps')
    args = parser.parse_args()
    train(load_config(args.config), args.arm, args.output, resume=args.resume,
          until_epoch=args.until_epoch, paired_steps=args.paired_steps, config_path=args.config)
