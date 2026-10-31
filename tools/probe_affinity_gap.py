# -*- coding: utf-8 -*-
"""冻结部署的只读逐阶段追踪；不修改融合、阈值、噪声或后处理。

用 --code-root 指定已有源码快照。仅在进程中挂载 decoder hook，保留每个
原生窗口的 CUT 概率、融合分量、插值边界及预定 ROI。输出必须放在私有目录。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def decompose(logits, options):
    """严格保留原 gated/top2 的运算顺序；网络预测 SAME，这里显式转 CUT。"""
    import torch
    import torch.nn.functional as F
    if logits.ndim != 4 or logits.shape[1] != 8:
        raise ValueError('Expected eight SAME-affinity logits')
    if options.get('short_reduction') != 'top2':
        raise ValueError('This trace is limited to the frozen top2 contract')
    same = torch.sigmoid(logits.float())
    cut = 1.0 - same
    short = cut[:, :4].topk(k=2, dim=1).values.mean(dim=1, keepdim=True)
    d2 = 1.0 - same[:, 4:6].mean(dim=1, keepdim=True)
    d4 = 1.0 - same[:, 6:8].mean(dim=1, keepdim=True)
    support2 = torch.sigmoid((F.max_pool2d(short, 5, 1, 2) - options['support_threshold']) / options['support_temperature'])
    support4 = torch.sigmoid((F.max_pool2d(short, 9, 1, 4) - options['support_threshold']) / options['support_temperature'])
    weight2 = float(options['distance2_weight']) * support2
    weight4 = float(options['distance4_weight']) * support4
    fused = ((short + weight2 * d2 + weight4 * d4) / (1.0 + weight2 + weight4)).clamp(0., 1.)
    return dict(cut=cut, short=short, distance2=d2, distance4=d4,
                support2=support2, support4=support4, weight2=weight2, weight4=weight4, fused=fused)


def blend_weights(height, width):
    wy = np.maximum(.05, 1 - np.abs(np.linspace(-1, 1, height, dtype=np.float32)))
    wx = np.maximum(.05, 1 - np.abs(np.linspace(-1, 1, width, dtype=np.float32)))
    return wy[:, None] * wx[None, :]


def fixed_roi(shape):
    """无私有诊断配置时，统一使用图像中心，不按图像身份挑选区域。"""
    height, width = shape
    size = min(512, height, width)
    cx = width // 2
    cy = height // 2
    left = max(0, min(width-size, cx-size//2))
    top = max(0, min(height-size, cy-size//2))
    return left, top, left+size, top+size


def load_roi_spec(path):
    """定位坐标来自私有 JSON，不能复制进可提交源码。"""
    if path is None:
        return {}
    value = json.loads(Path(path).read_text(encoding='utf-8'))
    if (not isinstance(value, dict) or value.get('format') != 'affinity_gap_roi_v1'
            or not isinstance(value.get('images'), dict)):
        raise ValueError('Expected affinity_gap_roi_v1 with an images object')
    return value['images']


def resolve_roi_spec(spec, stem, shape):
    height, width = shape
    if min(height, width) < 1:
        raise ValueError('Image dimensions must be positive')
    if stem not in spec:
        return fixed_roi(shape), dict(fixed_center=(width//2, height//2))
    row = spec[stem]
    if not isinstance(row, dict) or not isinstance(row.get('points'), dict) or not row['points']:
        raise ValueError('Each selected image needs roi and nonempty named points')
    roi = row.get('roi')
    if (not isinstance(roi, (list, tuple)) or len(roi) != 4
            or any(type(v) is not int for v in roi)):
        raise ValueError('ROI must contain four integer pixel coordinates')
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError('ROI is outside the source image or empty')
    points = {}
    for name, xy in row['points'].items():
        if (not isinstance(name, str) or not name or not isinstance(xy, (list, tuple))
                or len(xy) != 2 or any(type(v) is not int for v in xy)):
            raise ValueError('Points require a name and two integer pixel coordinates')
        x, y = xy
        if not (x0 <= x < x1 and y0 <= y < y1):
            raise ValueError('Diagnostic point is outside its ROI')
        points[name] = (x, y)
    return tuple(roi), points


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def render_gallery(folder, stem, rgb, roi, maps, points):
    """统一 0..1 色阶；只展示实际连续图，不画测试答案。"""
    import cv2
    from PIL import Image, ImageDraw
    x0, y0, x1, y1 = roi
    cells = [('RGB reference', rgb[y0:y1, x0:x1])]
    for name in ('short', 'distance2', 'distance4', 'fused', 'local_min', 'local_max'):
        heat = cv2.applyColorMap(np.rint(np.clip(maps[name], 0, 1)*255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        cells.append((name+' (0..1)', cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)))
    for threshold in (.45, .65):
        mask = (maps['fused'] > threshold).astype(np.uint8) * 255
        cells.append((f'actual fused > {threshold}', np.repeat(mask[..., None], 3, axis=2)))
    cell_width = 400
    cell_height = round((y1-y0)*cell_width/(x1-x0))
    canvas = Image.new('RGB', (3*cell_width, 45+3*(cell_height+28)), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f'{stem} fixed ROI {roi}; diagnostic probabilities, no GT', fill='black')
    for index, (title, value) in enumerate(cells):
        panel = Image.fromarray(value).resize((cell_width, cell_height), Image.Resampling.NEAREST)
        pd = ImageDraw.Draw(panel)
        for point_name, (px, py) in points.items():
            if x0 <= px < x1 and y0 <= py < y1:
                u, v = (px-x0)*cell_width/(x1-x0), (py-y0)*cell_height/(y1-y0)
                pd.line((u-5, v, u+5, v), fill='red', width=1)
                pd.line((u, v-5, u, v+5), fill='red', width=1)
        left, top = (index % 3)*cell_width, 45+(index//3)*(cell_height+28)
        draw.text((left+5, top+5), title, fill='black')
        canvas.paste(panel, (left, top+28))
    canvas.save(folder / f'{stem}_stages.png')


class Capture:
    def __init__(self, entry, config, destination, stem, shape, original, roi_spec=None):
        from utils.patch_diagnostic import BlendMap
        self.entry, self.config, self.destination = entry, config, destination
        self.stem, self.shape, self.original = stem, shape, original
        self.roi, self.points = resolve_roi_spec(roi_spec or {}, stem, shape)
        self.replay = BlendMap(shape)
        self.pending = None
        self.windows = []
        self.destination.mkdir(parents=True)
        x0, y0, x1, y1 = self.roi
        roi_shape = (y1-y0, x1-x0)
        self.roi_sum = {}
        self.roi_weight = np.zeros(roi_shape, np.float32)
        self.roi_min = np.full(roi_shape, np.inf, np.float32)
        self.roi_max = np.full(roi_shape, -np.inf, np.float32)

    def boundary(self, model, image, config):
        import torch
        from train_backend_adaptation import fusion_options
        captured = []
        def hook(_module, _arguments, output):
            captured.append(output['affinity_logits'].detach().float())
        handle = model.affinity_decoder.register_forward_hook(hook)
        try:
            actual = self.original(model, image, config)
        finally:
            handle.remove()
        if len(captured) != 1 or self.pending is not None:
            raise RuntimeError('Unexpected decoder/callback order')
        mode, options = fusion_options(config)
        if mode != 'gated':
            raise RuntimeError('This trace only supports the original gated contract')
        components = decompose(captured[0], options)
        if not torch.equal(components['fused'], actual):
            raise RuntimeError('Fusion decomposition changed the actual grid')
        self.pending = components
        return actual

    def __call__(self, *, source_shape, box, boundary, ph, pw, size, window, **_):
        import torch
        from utils.affinity_deployment import crop_letterbox_output
        if self.pending is None:
            raise RuntimeError('Missing captured logits')
        components, self.pending = self.pending, None
        y, x, y1, x1 = box
        height, width = y1-y, x1-x
        weights = blend_weights(height, width)
        self.replay.add(box, boundary)
        grid_arrays = {name: value.cpu()[0].numpy() for name, value in components.items()}
        np.savez_compressed(self.destination / f'window_{window:02d}.npz',
                            **grid_arrays, native_fused=boundary, box=np.asarray(box))
        row = dict(window=window, box=list(box), grid_shape=list(components['fused'].shape[-2:]),
                   fusion_grid_exact=True, points={})
        x0, y0, x2, y2 = self.roi
        left, top, right, bottom = max(x, x0), max(y, y0), min(x1, x2), min(y1, y2)
        overlap = right > left and bottom > top
        native_maps = {}
        for name, value in components.items():
            native = crop_letterbox_output(value, 1024, ph, pw, (height, width)).cpu()[0].numpy()
            native_maps[name] = native
            if name == 'fused' and not np.array_equal(native[0], boundary):
                raise RuntimeError('Native interpolation changed the actual boundary')
            if overlap:
                src = np.s_[top-y:bottom-y, left-x:right-x]
                dst = np.s_[top-y0:bottom-y0, left-x0:right-x0]
                if name not in self.roi_sum:
                    self.roi_sum[name] = np.zeros((native.shape[0], *self.roi_weight.shape), np.float32)
                self.roi_sum[name][:, dst[0], dst[1]] += native[:, src[0], src[1]] * weights[src][None]
        if overlap:
            self.roi_weight[dst] += weights[src]
            self.roi_min[dst] = np.minimum(self.roi_min[dst], boundary[src])
            self.roi_max[dst] = np.maximum(self.roi_max[dst], boundary[src])
        for point_name, (px, py) in self.points.items():
            if not (x <= px < x1 and y <= py < y1):
                continue
            lx, ly = px-x, py-y
            gh, gw = components['fused'].shape[-2:]
            content_h = max(1, round((1024-ph)*gh/1024))
            content_w = max(1, round((1024-pw)*gw/1024))
            gx = lx*(content_w-1)/max(1, width-1)
            gy = ly*(content_h-1)/max(1, height-1)
            bx, by = int(np.floor(gx)), int(np.floor(gy))
            row['points'][point_name] = dict(xy=[px, py], raw_weight=float(weights[ly, lx]),
                grid_xy=[gx, gy], values={name: arr[:, ly, lx].tolist() for name, arr in native_maps.items()},
                fused_grid_neighbors=grid_arrays['fused'][0, by:min(by+2, gh), bx:min(bx+2, gw)].tolist())
        self.windows.append(row)

    def finish(self, rgb, result, expected):
        replayed, _ = self.replay.finish()
        if not np.array_equal(replayed, result['boundary']) or not np.array_equal(replayed, expected):
            raise RuntimeError('BlendMap replay or N0 cached boundary is not exact')
        maps = {name: value/self.roi_weight[None] for name, value in self.roi_sum.items()}
        maps.update(local_min=self.roi_min[None], local_max=self.roi_max[None])
        x0, y0, x1, y1 = self.roi
        if not np.array_equal(maps['fused'][0], expected[y0:y1, x0:x1]):
            raise RuntimeError('ROI weighted replay differs')
        np.savez_compressed(self.destination / 'roi.npz', **maps, weight=self.roi_weight)
        points = {}
        for name, (px, py) in self.points.items():
            rows = [dict(window=w['window'], **w['points'][name]) for w in self.windows if name in w['points']]
            total = sum(r['raw_weight'] for r in rows)
            for row in rows:
                row['normalized_weight'] = row['raw_weight']/total
            points[name] = dict(xy=[px, py], stitched=float(expected[py, px]), windows=rows)
        render_gallery(self.destination.parent / 'gallery', self.stem, rgb, self.roi,
                       {name: value[0] for name, value in maps.items()}, self.points)
        return dict(source=self.stem, roi=list(self.roi), points=points, windows=self.windows,
                    fusion_grid_exact=True, native_interpolation_exact=True,
                    blend_replay_exact=True, n0_cached_boundary_exact=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--roi-spec', type=Path,
                        help='Private affinity_gap_roi_v1 JSON; omitted images use a generic center ROI')
    args = parser.parse_args()
    roi_spec = load_roi_spec(args.roi_spec)
    if args.output.exists():
        raise FileExistsError('Use a fresh output directory; do not overwrite prior evidence')
    sys.path.insert(0, str(args.code_root.resolve()))
    import cv2
    import torch
    from tools import infer_semifinal as entry
    from train_backend_adaptation import tensor_digest
    from utils.affinity_graph import DEFAULT_AFFINITY_OFFSETS
    baseline = json.loads((args.baseline / 'audit/report.json').read_text())
    for relative, expected in baseline['imported_sources'].items():
        if file_sha(args.code_root / relative) != expected:
            raise RuntimeError('Sealed source changed: '+relative)
    config = baseline['config']
    if Path(config['paths']['project_root']).resolve() != args.code_root.resolve():
        raise RuntimeError('Baseline source root differs')
    entry.validate_config(config)
    entry.configure_runtime()
    paths = entry.model_paths(config, args.model_dir.resolve())
    model, restorer, terminal = entry.load_models(config, paths, 'cuda')
    if terminal is not None or entry.local_restoration(config)['mode'] != 'single':
        raise RuntimeError('Expected original N0 only')
    before = dict(backend=tensor_digest(model.state_dict().items()), restorer=tensor_digest(restorer.state_dict().items()))
    if before != baseline['model_states']:
        raise RuntimeError('Loaded model state differs from the N0 baseline')
    args.output.mkdir(parents=True)
    (args.output / 'gallery').mkdir()
    report = dict(complete=False, training=False, gt_used=False, script_sha256=file_sha(__file__),
                  source_root=str(args.code_root), baseline=str(args.baseline), config=config,
                  roi_spec_sha256=file_sha(args.roi_spec) if args.roi_spec else None,
                  direction_offsets_yx=[list(v) for v in DEFAULT_AFFINITY_OFFSETS],
                  semantics='decoder sigmoid=SAME; CUT=1-sigmoid; diagnostic values are not accuracy',
                  fixed_sources=['test_116', 'test_101', 'test_166'], images=[])
    start = time.time()
    reference_rows = {row['source']: row for row in baseline['images']}
    for stem in report['fixed_sources']:
        matches = [p for p in args.input_dir.iterdir() if p.stem == stem]
        if len(matches) != 1:
            raise RuntimeError('Missing or duplicate source '+stem)
        path = matches[0]
        old = reference_rows[stem]
        if file_sha(path) != old['input_sha256']:
            raise RuntimeError('Input image identity differs')
        shape = cv2.imread(str(path)).shape[:2]
        capture = Capture(entry, config, args.output / stem, stem, shape, entry.views.boundary_grid, roi_spec)
        fixed_audit, local_audit = {}, []
        print('START', stem, flush=True)
        with patch.object(entry.views, 'boundary_grid', side_effect=capture.boundary), torch.no_grad():
            rgb, result = entry.predict(model, restorer, terminal, path, config, 'cuda',
                                       fixed_audit=fixed_audit, local_audit=local_audit, analysis_callback=capture)
        if fixed_audit != old['fixed_paths'] or local_audit != old['local_windows']:
            raise RuntimeError('Global path, window sequence, input or restored RGB differs')
        cached = np.load(args.baseline / 'analysis' / f'{stem}.npz')['boundary']
        row = capture.finish(rgb, result, cached)
        ids = cv2.imread(str(args.baseline / 'results' / f'{stem}_inst.png'), cv2.IMREAD_UNCHANGED)
        classes = json.loads((args.baseline / 'results' / f'{stem}_class.json').read_text())
        if not np.array_equal(ids, result['instances']) or classes != result['classes']:
            raise RuntimeError('Final N0 result changed')
        row.update(final_instances_exact=True, final_classes_exact=True,
                   fixed_paths_exact=True, all_window_noise_and_inputs_exact=True)
        report['images'].append(row)
        write_json(args.output / 'summary.json', report)
        print('EXACT', stem, len(capture.windows), 'windows', flush=True)
    after = dict(backend=tensor_digest(model.state_dict().items()), restorer=tensor_digest(restorer.state_dict().items()))
    if before != after:
        raise RuntimeError('Frozen model state changed')
    for relative, expected in baseline['imported_sources'].items():
        if file_sha(args.code_root / relative) != expected:
            raise RuntimeError('Sealed source changed during trace: '+relative)
    report.update(complete=True, frozen_model_states=before, source_unchanged=True,
                  elapsed_seconds=time.time()-start)
    write_json(args.output / 'summary.json', report)
    print('COMPLETE', report['elapsed_seconds'], flush=True)


if __name__ == '__main__':
    main()
