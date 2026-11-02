# -*- coding: utf-8 -*-
"""纯引擎C流程：全图恢复语义、局部affinity、NAF、原图投票。"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize
import torch
from torch.nn import functional as F

from .engine import Engine
from .postprocess import watershed_instances, vote_instances, merge_final_instances, verify_output


ROLES = ('trunk', 'semantic', 'affinity', 'naf', 'd5a')
FEATURES = ('f0', 'f1', 'f2', 'f3')


def engine_files(engine_dir, manifest=None):
    engine_dir = Path(engine_dir).resolve()
    path = Path(manifest).resolve() if manifest else engine_dir / 'engines.json'
    if path.is_file():
        document = json.loads(path.read_text(encoding='utf-8'))
        rows = document.get('engines', {})
        files = {}
        for role in ROLES:
            row = rows.get(role)
            if not isinstance(row, dict) or not isinstance(row.get('file'), str):
                raise ValueError('Engine manifest needs a file for role: ' + role)
            files[role] = (path.parent / row['file']).resolve()
    elif manifest:
        raise FileNotFoundError(path)
    else:
        files = {role:engine_dir / (role + '.engine') for role in ROLES}
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError('Engine file not found: ' + str(path))
    return files


def load_engines(files):
    return {role:Engine(files[role]) for role in ROLES}


def configure_runtime():
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)


def image_tensor(rgb, device):
    height, width = rgb.shape[:2]
    scale = 1024 / max(height, width)
    new_h, new_w = int(round(height * scale)), int(round(width * scale))
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    ph, pw = 1024 - new_h, 1024 - new_w
    padded = cv2.copyMakeBorder(resized, 0, ph, 0, pw, cv2.BORDER_REFLECT)
    tensor = torch.from_numpy(padded).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.
    return tensor, ph, pw


def crop_native(output, ph, pw, shape):
    height, width = output.shape[-2:]
    content_h = max(1, int(round((1024 - ph) * height / 1024)))
    content_w = max(1, int(round((1024 - pw) * width / 1024)))
    output = output[:, :, :content_h, :content_w]
    return F.interpolate(output, size=shape, mode='bilinear', align_corners=True)


def tile_boxes(shape, size, overlap):
    def starts(length):
        if length <= size:
            return [0]
        values = list(range(0, length - size + 1, size - overlap))
        if values[-1] != length - size:
            values.append(length - size)
        return values
    height, width = map(int, shape)
    return [(y, x, min(height, y + size), min(width, x + size))
            for y in starts(height) for x in starts(width)]


def weights(height, width):
    wy = np.maximum(.05, 1 - np.abs(np.linspace(-1, 1, height, dtype=np.float32)))
    wx = np.maximum(.05, 1 - np.abs(np.linspace(-1, 1, width, dtype=np.float32)))
    return wy[:, None] * wx[None, :]


def affinity_boundary(logits):
    probability = torch.sigmoid(logits)
    short = (1.0 - probability[:, :4]).topk(k=2, dim=1).values.mean(dim=1, keepdim=True)
    distance2 = 1.0 - probability[:, 4:6].mean(dim=1, keepdim=True)
    distance4 = 1.0 - probability[:, 6:8].mean(dim=1, keepdim=True)
    support2 = torch.sigmoid((F.max_pool2d(short, kernel_size=5, stride=1, padding=2) - .20) / .05)
    support4 = torch.sigmoid((F.max_pool2d(short, kernel_size=9, stride=1, padding=4) - .20) / .05)
    weighted2, weighted4 = float(.5) * support2, float(.25) * support4
    fused = (short + weighted2 * distance2 + weighted4 * distance4) / (1.0 + weighted2 + weighted4)
    return fused.clamp(0.0, 1.0)


def normalize_boundary(boundary):
    binary = np.ascontiguousarray(boundary >= .5, dtype=bool)
    skeleton = np.ascontiguousarray(skeletonize(binary), dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    normalized = cv2.dilate(skeleton, kernel, iterations=1, anchor=(-1, -1),
                            borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    return np.ascontiguousarray(normalized, dtype=bool)


class Pipeline:
    def __init__(self, engines, device='cuda'):
        self.engines, self.device = engines, torch.device(device)
        self.calls = {role:0 for role in ROLES}
        self._check_io()

    def _check_io(self):
        expected = dict(trunk={'image'}, semantic=set(FEATURES), affinity=set(FEATURES),
                        naf={'inputs'}, d5a={'xt', 'condition', 't_index'})
        required = dict(trunk=set(FEATURES), semantic={'semantic_logits'},
                        affinity={'affinity_logits'}, naf={'logits'}, d5a={'x0'})
        for role in ROLES:
            actual = set(self.engines[role].inputs)
            allowed = expected[role] | ({'image'} if role == 'semantic' else set())
            if not expected[role] <= actual <= allowed or not required[role] <= set(self.engines[role].outputs):
                raise ValueError('Engine I/O does not match role: ' + role)

    def _call(self, role, **inputs):
        self.calls[role] += 1
        return self.engines[role](**inputs)

    def _restore(self, condition):
        step = torch.full((condition.shape[0],), 16, dtype=torch.long, device=condition.device)
        raw = self._call('d5a', xt=condition, condition=condition, t_index=step)['x0'].float()
        # 原零噪terminal q_sample的state逐值等于condition；保留原finish算术。
        return (condition + 1.0 * (raw - condition)).clamp(0.0, 1.0)

    def _features(self, image):
        result = self._call('trunk', image=image)
        return {name:result[name] for name in FEATURES}

    def _semantic(self, image):
        inputs = self._features(image)
        if 'image' in self.engines['semantic'].inputs:
            inputs['image'] = image
        return self._call('semantic', **inputs)['semantic_logits']

    def _naf(self, boundary):
        source = normalize_boundary(boundary).astype(np.float32)
        padded = np.pad(source, 32, mode='reflect')
        boxes = tile_boxes(padded.shape, 256, 64)
        total = np.zeros(padded.shape, np.float32)
        weight_sum = np.zeros_like(total)
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=False):
            for start in range(0, len(boxes), 16):
                chunk = boxes[start:start + 16]
                inputs = []
                for y, x, y1, x1 in chunk:
                    value = padded[y:y1, x:x1]
                    height, width = value.shape
                    if (height, width) != (256, 256):
                        value = np.pad(value, ((0, 256 - height), (0, 256 - width)), mode='reflect')
                    inputs.append(value[None])
                tensor = torch.from_numpy(np.stack(inputs).copy()).to(self.device)
                prediction = self._call('naf', inputs=tensor)['logits'].sigmoid()
                if (tuple(prediction.shape) != (len(chunk), 1, 256, 256)
                        or not bool(torch.isfinite(prediction).all())):
                    raise RuntimeError('Invalid NAF output')
                arrays = prediction.detach().float().cpu().numpy()[:, 0]
                for output, (y, x, y1, x1) in zip(arrays, chunk):
                    height, width = y1 - y, x1 - x
                    weight = weights(height, width)
                    total[y:y1, x:x1] += output[:height, :width] * weight
                    weight_sum[y:y1, x:x1] += weight
        if not np.all(weight_sum > 0):
            raise RuntimeError('NAF tiles missed native pixels')
        result = (total / weight_sum)[32:-32, 32:-32]
        return np.clip(result, 0, 1).astype(np.float32)

    @torch.no_grad()
    def predict(self, rgb):
        shape = rgb.shape[:2]
        whole, ph, pw = image_tensor(rgb, self.device)
        restored_semantic = self._semantic(self._restore(whole))
        semantic = crop_native(restored_semantic, ph, pw, shape).cpu()
        raw_semantic = self._semantic(whole)
        raw_probability = crop_native(raw_semantic.float(), ph, pw, shape).cpu().sigmoid()[0, 0].numpy()

        total = np.zeros(shape, np.float32)
        weight_sum = np.zeros(shape, np.float32)
        for y, x, y1, x1 in tile_boxes(shape, 1024, 256):
            image, local_ph, local_pw = image_tensor(rgb[y:y1, x:x1], self.device)
            restored = self._restore(image)
            outputs = self._call('affinity', **self._features(restored))
            boundary = affinity_boundary(outputs['affinity_logits'].float())
            native = crop_native(boundary, local_ph, local_pw, (y1 - y, x1 - x)).cpu()[0, 0].numpy()
            if not np.isfinite(native).all():
                raise RuntimeError('Invalid affinity boundary')
            weight = weights(y1 - y, x1 - x)
            total[y:y1, x:x1] += native * weight
            weight_sum[y:y1, x:x1] += weight
        if not np.all(weight_sum > 0):
            raise RuntimeError('Native tiles missed image pixels')
        before = total / weight_sum
        after = self._naf(before)
        ids = watershed_instances(semantic, after, before)
        classes = vote_instances(ids, raw_probability)
        ids, classes = merge_final_instances(ids, classes, 200)
        verify_output(ids, classes, shape)
        return ids, classes


def input_files(path):
    path = Path(path)
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted((file for file in path.iterdir() if file.is_file() and file.suffix.lower() in extensions),
                       key=lambda file:file.name)
    else:
        raise FileNotFoundError(path)
    if not files or len({file.stem for file in files}) != len(files):
        raise ValueError('Input needs readable image files with distinct stems')
    return files


def read_rgb(path):
    image = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Could not read image: ' + str(path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_prediction(folder, stem, ids, classes):
    verify_output(ids, classes, ids.shape)
    success, encoded = cv2.imencode('.png', ids)
    if not success:
        raise RuntimeError('Could not encode result PNG')
    (folder / (stem + '_inst.png')).write_bytes(encoded.tobytes())
    (folder / (stem + '_class.json')).write_text(
        json.dumps(classes, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
