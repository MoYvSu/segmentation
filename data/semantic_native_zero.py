# -*- coding: utf-8 -*-
"""零噪语义适配的原生部署网格；这里只读训练原图与既有 radius8 GT。

每源的部署窗按独立随机流循环排列，每轮取四窗。D4、源图顺序及窗序均由
epoch/index直接确定，不消耗全局随机流，恢复训练时无需重放前面的读取。
本模块不运行 D5a；返回的原图应由训练器在 D4 之后送入冻结零噪 D5a。
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from random import Random
import zipfile

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from data.dataset import letterbox
from data.direct_dual_head_dataset import _spatial_transform
from data.rgb_restoration_dataset import read_rgb
from data.semantic_targets import semantic_targets_from_instances
from utils.patch_diagnostic import tile_boxes


FORMAT = 'semantic_native_zero_dataset_v1'
MAX_SOURCE_PIXELS = 6_000_000
ARRAY_KEYS = ('instance_map', 'original_covered', 'filled', 'residual_unknown')


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode('utf-8')).hexdigest()


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=_unique_pairs)


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


def _seed(seed, *parts):
    return int(_digest([seed, *parts])[:16], 16)


def _binary(value, shape, name):
    if value.shape != shape or value.dtype.kind not in 'buif' or not np.all((value == 0) | (value == 1)):
        raise ValueError(f'{name} must be a binary mask with source shape')
    # 后续逻辑一律用 bool；uint8 mask 不能用作 NumPy 位置索引。
    return value.astype(bool, copy=False)


def _check_npz_headers(path, shape):
    """先检查解压后 shape/dtype，避免畸形压缩成员先分配大数组。"""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError(f'Duplicate NPZ members: {path}')
        for key in ARRAY_KEYS:
            name = key + '.npy'
            if name not in names:
                raise ValueError(f'Missing GT array {key}: {path}')
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    array_shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    array_shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError(f'Unsupported NPY header version: {version}')
                if tuple(array_shape) != shape or dtype.hasobject or dtype.kind not in 'buif':
                    raise ValueError(f'GT array shape/dtype differs from source: {key}')
                expected = stream.tell() + int(np.prod(shape)) * dtype.itemsize
                if archive.getinfo(name).file_size != expected:
                    raise ValueError(f'GT array payload length mismatch: {key}')


def _boxed(value, content_shape, size, fill, dtype):
    height, width = content_shape
    result = np.full((size, size), fill, dtype=dtype)
    if value.shape != (height, width):
        value = cv2.resize(value.astype(np.float32) if value.dtype == bool else value,
                           (width, height), interpolation=cv2.INTER_NEAREST)
    result[:height, :width] = value
    return result


class NativeSemanticDataset(Dataset):
    """单样本、无大数组缓存；默认32源/12窗/每轮4窗/20轮。

    小尺寸参数只用于合成 CPU 检查；正式入口应固定 native_size=1024、
    overlap=.25、expected_tiles_per_source=12。LabelMe仅核身份，不读取其多边形。
    """

    def __init__(self, raw_data_dir, completed_gt_dir, *, seed, expected_sources=32,
                 epochs=20, draws_per_source=4, monitor_sources=None,
                 native_size=1024, overlap=.25, expected_tiles_per_source=12):
        self.raw_data_dir = Path(raw_data_dir).resolve()
        self.completed_gt_dir = Path(completed_gt_dir).resolve()
        self.seed = _integer(seed, 'seed', 0)
        self.expected_sources = _integer(expected_sources, 'expected_sources')
        self.epochs = _integer(epochs, 'epochs')
        self.draws_per_source = _integer(draws_per_source, 'draws_per_source')
        self.native_size = _integer(native_size, 'native_size')
        self.expected_tiles_per_source = _integer(expected_tiles_per_source, 'expected_tiles_per_source')
        if isinstance(overlap, bool) or not isinstance(overlap, (int, float)) or not 0 <= overlap < 1:
            raise ValueError('overlap must be in [0,1)')
        self.overlap = float(overlap)
        if self.expected_tiles_per_source % self.draws_per_source:
            raise ValueError('draws_per_source must divide the deployment tile count')
        self.manifest_path = self.completed_gt_dir / 'manifest.json'
        manifest = _json(self.manifest_path)
        self.manifest_sha256 = _sha(self.manifest_path)
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        paths = sorted(p for p in self.raw_data_dir.iterdir()
                       if p.suffix.lower() in extensions and p.with_suffix('.json').is_file())
        stems = [p.stem for p in paths]
        if len(paths) != self.expected_sources or len(set(stems)) != len(stems):
            raise ValueError('Manual image/JSON cohort differs from the requested source count')
        if (manifest.get('format') != 'manual_target_v2' or manifest.get('version') != 1
                or manifest.get('sample_count') != self.expected_sources
                or sorted(manifest.get('samples', [])) != sorted(stems)
                or manifest.get('max_fill_distance_native') != 8.
                or manifest.get('blur_sigma') != 1.2):
            raise ValueError('Approved manual_target_v2/v1 radius8 manifest differs')
        for suffix in ('_gt.npz', '_class.json'):
            if {p.name for p in self.completed_gt_dir.glob('*' + suffix)} != {s + suffix for s in stems}:
                raise ValueError(f'GT file cohort differs from the exact manual sources: {suffix}')
        self.sources = []
        for path in paths:
            with Image.open(path) as image:
                width, height = image.size
            shape = (height, width)
            if min(shape) <= 0 or height * width > MAX_SOURCE_PIXELS:
                raise ValueError(f'Source shape exceeds the bounded native contract: {path.name}')
            labelme = _json(path.with_suffix('.json'))
            if (not isinstance(labelme, dict)
                    or ('imagePath' in labelme and str(labelme['imagePath']).replace('\\', '/').split('/')[-1] != path.name)
                    or labelme.get('imageHeight', height) != height or labelme.get('imageWidth', width) != width):
                raise ValueError(f'LabelMe identity differs from source image: {path.name}')
            boxes = tile_boxes(shape, self.native_size, self.overlap)
            if len(boxes) != self.expected_tiles_per_source:
                raise ValueError(f'Deployment tile count differs for {path.name}: {len(boxes)}')
            files = dict(image=path, labelme=path.with_suffix('.json'),
                         gt=self.completed_gt_dir / (path.stem + '_gt.npz'),
                         classes=self.completed_gt_dir / (path.stem + '_class.json'))
            self.sources.append(dict(source=path.stem, shape=list(shape), boxes=[list(b) for b in boxes],
                files={k: dict(path=str(p), name=p.name, sha256=_sha(p)) for k, p in files.items()}))
        sealed = [dict(source=r['source'], shape=r['shape'], boxes=r['boxes'],
                       files={k: dict(name=f['name'], sha256=f['sha256']) for k, f in r['files'].items()})
                  for r in self.sources]
        self.source_seal_sha256 = _digest(dict(manifest_sha256=self.manifest_sha256, sources=sealed))
        if monitor_sources is None:
            monitor_sources = (stems[:3] + stems[-1:]) if len(stems) >= 4 else stems
        self.monitor_sources = list(monitor_sources)
        if len(set(self.monitor_sources)) != len(self.monitor_sources) or not set(self.monitor_sources) <= set(stems):
            raise ValueError('Monitor sources must be distinct members of the manual cohort')
        self.epoch = 1

    def __len__(self):
        return self.expected_sources * self.draws_per_source

    def set_epoch(self, epoch):
        epoch = _integer(epoch, 'epoch')
        if epoch > self.epochs:
            raise ValueError('epoch exceeds the prespecified training budget')
        self.epoch = epoch

    def _verify_source(self, row):
        if _sha(self.manifest_path) != self.manifest_sha256:
            raise RuntimeError('GT manifest changed after dataset sealing')
        for key, record in row['files'].items():
            if _sha(record['path']) != record['sha256']:
                raise RuntimeError(f'Sealed source changed: {row["source"]}/{key}')

    def verify_sources(self):
        for row in self.sources:
            self._verify_source(row)
        return True

    def identity(self):
        self.verify_sources()
        return dict(format=FORMAT, manifest_sha256=self.manifest_sha256,
                    source_seal_sha256=self.source_seal_sha256, source_count=self.expected_sources,
                    seed=self.seed, epochs=self.epochs, draws_per_source=self.draws_per_source,
                    updates_per_epoch=len(self), total_updates=len(self) * self.epochs,
                    native_size=self.native_size, overlap=self.overlap,
                    expected_tiles_per_source=self.expected_tiles_per_source,
                    cycles_per_source=(self.epochs * self.draws_per_source) / self.expected_tiles_per_source,
                    supervision='completed GT original+filled equal weight; zero ID/padding ignored',
                    boundary='native GT transitions before crop; no artificial crop/padding frame',
                    augmentation='D4 only, before frozen zero-noise D5a; no degradation',
                    cache_full_arrays=False)

    def source_seal(self):
        self.verify_sources()
        return dict(format=FORMAT, source_seal_sha256=self.source_seal_sha256,
                    manifest_sha256=self.manifest_sha256, sources=deepcopy(self.sources))

    def monitor_manifest(self):
        by_name = {r['source']: r for r in self.sources}
        return [dict(source=s, crop_box=by_name[s]['boxes'][0], rotation_k=0,
                     horizontal_flip=False, vertical_flip=False, identity_D4=True,
                     files=deepcopy(by_name[s]['files'])) for s in self.monitor_sources]

    def manifest(self):
        return dict(**self.identity(), sources=deepcopy(self.sources),
                    monitor=self.monitor_manifest(),
                    sampling='source order shuffled by epoch; per-source deployment windows independently shuffled each cycle')

    def sample_spec(self, index):
        index = _integer(index, 'index', 0)
        if index >= len(self):
            raise IndexError(index)
        order = list(range(self.expected_sources))
        Random(_seed(self.seed, 'source_order', self.epoch)).shuffle(order)
        source_index = order[index // self.draws_per_source]
        local_draw = index % self.draws_per_source
        source_draw = (self.epoch - 1) * self.draws_per_source + local_draw
        cycle, offset = divmod(source_draw, self.expected_tiles_per_source)
        row = self.sources[source_index]
        windows = list(range(self.expected_tiles_per_source))
        Random(_seed(self.seed, 'window_cycle', row['source'], cycle)).shuffle(windows)
        window_index = windows[offset]
        transform = Random(_seed(self.seed, 'D4', row['source'], source_draw)).randrange(8)
        return dict(source=row['source'], source_index=source_index, epoch=self.epoch,
                    draw_index=local_draw, global_draw=(self.epoch - 1) * len(self) + index,
                    source_draw=source_draw, window_index=window_index, crop_box=row['boxes'][window_index],
                    rotation_k=transform // 2, horizontal_flip=bool(transform % 2), vertical_flip=False,
                    d4_index=transform)

    def _load(self, spec):
        row = self.sources[spec['source_index']]
        self._verify_source(row)
        shape = tuple(row['shape'])
        files = row['files']
        rgb = read_rgb(files['image']['path'])
        if rgb.shape != (*shape, 3) or rgb.dtype != np.uint8:
            raise ValueError('Decoded RGB differs from the sealed source shape/type')
        _check_npz_headers(files['gt']['path'], shape)
        with np.load(files['gt']['path'], allow_pickle=False) as blob:
            ids = blob['instance_map']
            if ids.dtype.kind not in 'iu' or np.any(ids < 0) or int(ids.max()) > 65535:
                raise ValueError('GT instance IDs must be nonnegative integers <=65535')
            ids = ids.astype(np.int32, copy=False)
            original = _binary(blob['original_covered'], shape, 'original_covered')
            filled = _binary(blob['filled'], shape, 'filled')
            unknown = _binary(blob['residual_unknown'], shape, 'residual_unknown')
        if (not np.array_equal(unknown, ids == 0) or np.any(original & filled)
                or not np.array_equal(original | filled, ids > 0)):
            raise ValueError('GT provenance masks disagree with completed instance IDs')
        classes = _json(files['classes']['path'])
        present = set(int(x) for x in np.unique(ids) if x > 0)
        if (not isinstance(classes, dict) or set(classes) != {str(i) for i in present}
                or any(type(v) is not int or v not in (0, 1) for v in classes.values())):
            raise ValueError('Class mapping must contain exactly the positive GT IDs with integer P0/F1')
        lookup = np.full(int(ids.max()) + 1, -1, np.float32)
        for key, value in classes.items():
            lookup[int(key)] = value
        y0, x0, y1, x1 = spec['crop_box']
        # 一像素原图 halo 足够保留真正的四邻接 GT 转换；不把裁窗边补成零 ID。
        hy, hx, ey, ex = max(0, y0 - 1), max(0, x0 - 1), min(shape[0], y1 + 1), min(shape[1], x1 + 1)
        semantic, boundary = semantic_targets_from_instances(ids[hy:ey, hx:ex], lookup)
        cut = np.s_[y0-hy:y1-hy, x0-hx:x1-hx]
        semantic, boundary = semantic[cut], boundary[cut]
        crop_ids = ids[y0:y1, x0:x1]
        counts = np.bincount(ids.reshape(-1), minlength=len(lookup))
        visible_counts = np.bincount(crop_ids.reshape(-1), minlength=len(lookup))
        visible_ids = np.flatnonzero(visible_counts[1:] > 0) + 1
        truncated = int(np.count_nonzero(visible_counts[visible_ids] < counts[visible_ids]))
        image, _, pad_h, pad_w = letterbox(rgb[y0:y1, x0:x1], self.native_size)
        content_shape = (self.native_size - pad_h, self.native_size - pad_w)
        boxed_ids = _boxed(crop_ids, content_shape, self.native_size, 0, np.int32)
        semantic = _boxed(semantic, content_shape, self.native_size, -1, np.float32)
        boundary = _boxed(boundary, content_shape, self.native_size, 0, np.float32)
        content = _boxed(np.ones(crop_ids.shape, bool), content_shape, self.native_size, False, bool)
        original_box = _boxed(original[y0:y1, x0:x1], content_shape, self.native_size, False, bool)
        filled_box = _boxed(filled[y0:y1, x0:x1], content_shape, self.native_size, False, bool)
        valid = content & (boxed_ids > 0)
        if not np.array_equal(semantic >= 0, valid) or not np.array_equal(original_box | filled_box, valid):
            raise RuntimeError('Boxing lost the GT/ignore alignment')
        known = int(np.count_nonzero(valid))
        if known == 0:
            raise ValueError(f'Deployment training crop has no known semantic supervision: {spec}')
        ledger = dict(known_pixels=known, unknown_pixels=int(np.count_nonzero(semantic < 0)),
                      original_covered_pixels=int(np.count_nonzero(original_box)),
                      filled_pixels=int(np.count_nonzero(filled_box)),
                      padding_pixels=int(np.count_nonzero(~content)),
                      native_known_pixels=int(np.count_nonzero(crop_ids > 0)),
                      native_unknown_pixels=int(np.count_nonzero(crop_ids == 0)),
                      visible_instances=int(len(visible_ids)), truncated_instances=truncated,
                      visible_F_instances=int(np.count_nonzero(lookup[visible_ids] == 1)),
                      visible_P_instances=int(np.count_nonzero(lookup[visible_ids] == 0)))
        tensors = dict(image=torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float() / 255.,
                       semantic_target=torch.from_numpy(semantic)[None],
                       semantic_boundary=torch.from_numpy(boundary)[None],
                       semantic_instance_map=torch.from_numpy(boxed_ids.astype(np.int64)),
                       semantic_valid_content=torch.from_numpy(valid)[None],
                       valid_content=torch.from_numpy(content)[None])
        tensors = {k: _spatial_transform(v, spec['horizontal_flip'], spec['vertical_flip'], spec['rotation_k'])
                   for k, v in tensors.items()}
        return dict(**tensors, **spec, name=spec['source'], image_name=Path(files['image']['path']).name,
                    image_path=files['image']['path'], ledger=ledger,
                    known_pixels=known, truncated_instances=truncated)

    def __getitem__(self, index):
        return self._load(self.sample_spec(index))

    def monitor_samples(self):
        by_name = {r['source']: i for i, r in enumerate(self.sources)}
        return [self._load(dict(source=s, source_index=by_name[s], epoch=0, draw_index=-1,
                    global_draw=-1, source_draw=-1, window_index=0, crop_box=self.sources[by_name[s]]['boxes'][0],
                    rotation_k=0, horizontal_flip=False, vertical_flip=False, d4_index=0))
                for s in self.monitor_sources]
