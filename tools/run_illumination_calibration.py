# -*- coding: utf-8 -*-
"""校光实验入口：核对既有两头、短测、训练、固定四图前后位置对照；不打包。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from data.mim_dataset import list_images
from models.backend_adaptation import load_backend, restore_first
from models.illumination_calibration import IlluminationCalibrator
from tools.semantic_crossover import revote_instances
from train_backend_adaptation import fusion_options, save_prediction, sha, tensor_digest, write_json
from train_semantic_d5a import get_restorer
from utils.affinity_deployment import prepare_image, crop_letterbox_output, crop_affinity_boundary_output
from utils.config import load_config, project_path


VARIANTS = ('raw', 'light', 'd5a', 'light_d5a', 'd5a_light')


def same_state(actual, expected, name):
    if set(actual) != set(expected) or any(not torch.equal(actual[k].cpu(), expected[k].cpu()) for k in actual):
        raise RuntimeError(f'{name} differs from the designated source')


def verified_models(config, device):
    cfg = config['illumination_calibration']
    source = cfg['backend']
    path = Path(project_path(config, source['checkpoint']))
    if sha(path) != source['sha256']:
        raise RuntimeError('The designated original simple e60 checkpoint changed')
    backend, saved = load_backend(str(path), config, device)
    if (saved['epoch'] != source['simple_epoch'] or
            saved['backend_adaptation'].get('semantic_adaptation', {}).get('arm') != 'simple' or
            backend.semantic_decoder.semantic_residual is not None):
        raise RuntimeError('Expected the original simple head, not full/light/cold weights')
    affinity_path = Path(project_path(config, config['affinity_deployment']['checkpoint']))
    reference_path = Path(project_path(config, config['affinity_geometry_g1']['reference_checkpoint']))
    if sha(affinity_path) != source['affinity_sha256'] or sha(reference_path) != source['reference_sha256']:
        raise RuntimeError('Historical affinity/reference checkpoint changed')
    affinity = torch.load(affinity_path, map_location='cpu', weights_only=False)
    reference = torch.load(reference_path, map_location='cpu', weights_only=False)
    if affinity['epoch'] != source['affinity_epoch']:
        raise RuntimeError('Expected V affinity epoch 115')
    same_state(backend.affinity_decoder.state_dict(), affinity['geometry_state_dict'], 'V affinity tensors')
    lora = {k: v for k, v in backend.encoder.trunk.state_dict().items() if 'lora_A' in k or 'lora_B' in k}
    same_state(lora, reference['lora_state_dict'], 'joint-v3 LoRA tensors')
    restorer = get_restorer(config, device)
    if restorer.num_steps != 16 or restorer.kappa != .03:
        raise RuntimeError('Expected the designated D5a first-step recipe')
    for module in (backend, restorer):
        module.eval().requires_grad_(False)
    parameters = sum(p.numel() for m in (backend, restorer) for p in m.parameters())
    parameters += sum(p.numel() for p in IlluminationCalibrator(**cfg['model']).parameters())
    if parameters >= 500_000_000:
        raise RuntimeError('Combined model exceeds competition parameter limit')
    report = dict(simple=dict(path=str(path), epoch=saved['epoch'], sha256=source['sha256']),
                  affinity=dict(path=str(affinity_path), epoch=affinity['epoch'], sha256=sha(affinity_path), tensors_equal=True),
                  reference=dict(path=str(reference_path), epoch=reference['epoch'], sha256=sha(reference_path), lora_equal=True),
                  d5a=config['semantic_adaptation']['restoration'], total_parameters=parameters,
                  backend_state=tensor_digest(backend.state_dict().items()),
                  d5a_state=tensor_digest(restorer.state_dict().items()), strict_simple_load=True)
    return backend, restorer, report


def reflect_content(image, h, w):
    """与OpenCV BORDER_REFLECT一致；校光后重新反射真实内容。"""
    def axis(length, valid_length):
        idx = torch.arange(length, device=image.device) % (2 * valid_length)
        return torch.where(idx < valid_length, idx, 2 * valid_length - 1 - idx)
    return image[..., :h, :w].index_select(-2, axis(image.shape[-2], h)).index_select(-1, axis(image.shape[-1], w))


def calibrated(calibrator, image, ph, pw):
    valid = torch.zeros_like(image[:, :1])
    h, w = image.shape[-2] - ph, image.shape[-1] - pw
    valid[..., :h, :w] = 1
    # 反射校正增量，保留D5a已有的padding；恒等校光必须在整张tensor上逐值不变。
    delta = calibrator(image, valid) - image
    result = (image + reflect_content(delta, h, w)).clamp(0, 1)
    # 保持原部署tensor的完整stride（含B=1时的特殊batch stride），避免切换CUDA计算路径。
    return torch.empty_strided(image.shape, image.stride(), device=image.device, dtype=image.dtype).copy_(result)


@torch.no_grad()
def ordered_inputs(image, ph, pw, calibrator, restorer, seed):
    light = calibrated(calibrator, image, ph, pw)
    d5a = restore_first(restorer, image, seed)
    # 前后两臂共享校光权重、D5a权重和采样噪声seed。
    return dict(raw=image, light=light, d5a=d5a,
                light_d5a=restore_first(restorer, light, seed),
                d5a_light=calibrated(calibrator, d5a, ph, pw))


def thumb(image, title, width):
    image = cv2.resize(image, (width, round(image.shape[0] * width / image.shape[1])), interpolation=cv2.INTER_AREA)
    image = cv2.copyMakeBorder(image, 25, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(image, title, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1, cv2.LINE_AA)
    return image


def save_rgb(path, rgb):
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise IOError(f'Cannot write image: {path}')


def instance_overlay(image, instances, classes):
    lookup = np.zeros((int(instances.max()) + 1, 3), np.uint8)
    for key, value in classes.items():
        lookup[int(key)] = [245, 190, 55] if value == 1 else [65, 145, 235]
    output = np.where((instances > 0)[..., None], .64 * image + .36 * lookup[instances], image).astype(np.uint8)
    edges = np.zeros(instances.shape, bool)
    edges[1:] |= instances[1:] != instances[:-1]
    edges[:, 1:] |= instances[:, 1:] != instances[:, :-1]
    output[edges] = [200, 50, 210]
    return output


@torch.no_grad()
def render_comparison(config, calibrator, backend, restorer, directory, epoch, *, final=False, verify=False):
    cfg = config['illumination_calibration']
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    device = next(backend.parameters()).device
    backend.eval(); restorer.eval(); calibrator.eval()
    files = [Path(p) for p in list_images(project_path(config, config['inference']['test_dir']))]
    selected = [(i, p) for i, p in enumerate(files) if p.stem in cfg['monitor']['images']]
    if len(selected) != len(cfg['monitor']['images']):
        raise RuntimeError('Missing configured monitor image')
    summaries = []
    baseline_dir = Path(project_path(config, cfg['backend']['baseline_dir']))
    for index, path in selected:
        original, image, ph, pw = prepare_image(path, 1024, device)
        fixed = cv2.imread(str(baseline_dir / f'{path.stem}_inst.png'), cv2.IMREAD_UNCHANGED)
        if fixed is None or fixed.dtype != np.uint16 or fixed.shape != original.shape[:2]:
            raise RuntimeError('Missing original simple fixed instances')
        previous = json.loads((baseline_dir / f'{path.stem}_class.json').read_text())
        inputs = ordered_inputs(image, ph, pw, calibrator, restorer,
                                config['semantic_adaptation']['restoration']['inference_seed'] + index)
        rows = [[], [], [], []]
        predictions, boundaries, deployments, entries, final_panels = {}, {}, {}, {}, []
        for name, changed in inputs.items():
            result = backend(changed)
            sem = crop_letterbox_output(result['semantic_logits'], 1024, ph, pw, original.shape[:2]).cpu()
            mode, kwargs = fusion_options(config)
            edge = crop_affinity_boundary_output(result, 1024, ph, pw, original.shape[:2], mode, kwargs).cpu()
            prob = sem.sigmoid()[0, 0].numpy()
            classes, scores = revote_instances(fixed, prob)
            if name == 'd5a' and classes != previous:
                raise RuntimeError(f'{path.stem}: original simple class votes no longer reproduce')
            if verify:
                if name == 'light':
                    if not torch.equal(changed, image):
                        raise RuntimeError('Identity calibration changes the raw input/padding')
                if name in ('light_d5a', 'd5a_light'):
                    if not torch.equal(changed, inputs['d5a']):
                        raise RuntimeError('Identity calibration changes D5a input/padding')
            restored = crop_letterbox_output(changed, 1024, ph, pw, original.shape[:2])[0].permute(1, 2, 0).cpu().numpy()
            restored = np.rint(np.clip(restored, 0, 1) * 255).astype(np.uint8)
            probability = cv2.cvtColor(cv2.applyColorMap(np.rint(prob * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
            heat = cv2.cvtColor(cv2.applyColorMap(np.rint(edge[0, 0].numpy() * 255).astype(np.uint8), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)
            overlay = instance_overlay(original, fixed, classes)
            for row, content, label in zip(rows, (restored, probability, heat, overlay),
                                           (name, 'P(F) 0..1', 'Affinity 0..1', 'Fixed instances F=gold')):
                row.append(thumb(content, label, cfg['monitor']['thumbnail_size']))
            entries[name] = dict(ferrite=sum(c == 1 for c in classes.values()),
                                 flips_vs_d5a=sum(classes[k] != previous[k] for k in previous),
                                 mean_probability=float(prob.mean()), classes=classes, instance_scores=scores)
            predictions[name] = prob
            boundaries[name] = edge
            if verify and name in ('light', 'light_d5a', 'd5a_light'):
                parent = 'raw' if name == 'light' else 'd5a'
                if not np.array_equal(prob, predictions[parent]):
                    raise RuntimeError(f'Identity semantic mismatch {name}/{parent}: '
                        f'max={np.abs(prob-predictions[parent]).max()} '
                        f'input_strides={changed.stride()}/{inputs[parent].stride()}')
                if not torch.equal(edge, boundaries[parent]):
                    raise RuntimeError('Identity calibration changes affinity output')
            if final:
                out = directory / name
                out.mkdir(exist_ok=True)
                instances, deployed_classes = save_prediction(original, sem, edge, config, out, path.stem)
                if instances.dtype != np.uint16 or int(instances.max()) > 65535:
                    raise RuntimeError('Invalid final instance format')
                deployed_classes = {str(k): int(v) for k, v in deployed_classes.items()}
                if name == 'd5a' and (not np.array_equal(instances, fixed) or deployed_classes != previous):
                    raise RuntimeError('Full original simple deployment does not reproduce')
                deployments[name] = (instances, deployed_classes)
                if verify and name in ('light', 'light_d5a', 'd5a_light'):
                    parent_inst, parent_class = deployments['raw' if name == 'light' else 'd5a']
                    if not np.array_equal(instances, parent_inst) or deployed_classes != parent_class:
                        raise RuntimeError('Identity calibration changes complete final instances/classes')
                deployed_overlay = instance_overlay(original, instances, {str(k): int(v) for k, v in deployed_classes.items()})
                save_rgb(out / f'{path.stem}_input.png', restored)
                save_rgb(out / f'{path.stem}_overlay.png', deployed_overlay)
                np.savez_compressed(out / f'{path.stem}_maps.npz', ferrite_probability=prob, boundary=edge[0, 0].numpy())
                entries[name]['final_instances'] = len(deployed_classes)
                final_panels.append(thumb(deployed_overlay, name, 320))
        save_rgb(directory / f'{path.stem}.png', np.concatenate([np.concatenate(row, 1) for row in rows], 0))
        if final:
            save_rgb(directory / f'{path.stem}_instances.png', np.concatenate(final_panels, 1))
        delta = predictions['light_d5a'] - predictions['d5a_light']
        summaries.append(dict(image=path.name, epoch=epoch, variants=entries,
                              before_after_probability_mae=float(np.abs(delta).mean())))
    write_json(directory / 'report.json', dict(epoch=epoch, images=summaries,
        variants=list(VARIANTS), fixed_geometry='original simple e60 final instances',
        scope='Unlabelled image/model-response diagnostics; no official score or package', final_deployment=final))
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/train/illumination_calibration.yaml')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--short', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    cfg = config['illumination_calibration']
    output = Path(project_path(config, cfg['output_dir']))
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4); cv2.setNumThreads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('Use the GPU server sam2_env')
    backend, restorer, receipt = verified_models(config, torch.device('cuda'))
    write_json(output / 'sources.json', receipt)
    if args.preflight:
        init = IlluminationCalibrator(**cfg['model']).cuda().eval()
        render_comparison(config, init, backend, restorer, output / 'initial', 0, verify=True, final=True)
        write_json(output / 'preflight.json', dict(passed=True, sources=receipt))
        print('Preflight passed: designated simple/V, source LoRA, complete original deployment and identity content', flush=True)
        return
    from train_illumination_calibration import run_short_test, train
    if args.short:
        run_short_test(config, output / 'short')
        return
    preflight = json.loads((output / 'preflight.json').read_text())
    short = json.loads((output / 'short' / 'report.json').read_text())
    if not preflight['passed'] or not short['passed'] or preflight['sources'] != receipt:
        raise RuntimeError('Preflight/short gate/source consistency failed')
    started = time.time()
    status = dict(status='running', started_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                  config=config, official_submission=False)
    write_json(output / 'status.json', status)
    try:
        candidate = train(config, output, backend, restorer)
        render_comparison(config, candidate, backend, restorer, output / 'comparison', cfg['epochs'], final=True)
        if tensor_digest(backend.state_dict().items()) != receipt['backend_state'] or tensor_digest(restorer.state_dict().items()) != receipt['d5a_state']:
            raise RuntimeError('Frozen backend/D5a changed')
        status.update(status='completed', elapsed_seconds=time.time() - started,
                      frozen_weights_unchanged=True, final_epoch=cfg['epochs'])
    except Exception as error:
        status.update(status='failed', error=repr(error))
        raise
    finally:
        write_json(output / 'status.json', status)


if __name__ == '__main__':
    main()
