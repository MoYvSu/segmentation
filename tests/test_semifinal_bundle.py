# -*- coding: utf-8 -*-
"""用合成 checkpoint/100 对结果检查精确导出与完整目录包，不读赛题数据。"""
from copy import deepcopy
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import pytest
import torch
import yaml

from tools.build_semifinal_bundle import (
    MODEL_NAMES, NAMES, REQUIRED_PACKAGES, build_code, build_models, export_checkpoint,
    pack_submission, pinned_requirements, safe_relative, sha, source_rows,
    state_digest, validate_code_model, validate_results, write_json,
)


def checkpoints(tmp_path):
    architecture = dict(sam2=dict(config_file='configs/sam2.1/sam2.1_hiera_t.yaml',
                                  sam2_repo_path='/original/code/segment-anything-2',
                                  input_normalization='legacy_none'),
                        semantic_decoder=dict(fpn_channels=2))
    head = {'projection.weight': torch.tensor([[1.25, -0.375]], dtype=torch.float32)}
    backend_state = {'encoder.weight': torch.arange(4, dtype=torch.float32),
                     **{'semantic_decoder.' + key: value for key, value in head.items()}}
    backend = dict(format='phase_affinity_fused_v1', architecture=architecture, epoch=20,
                   model_state_dict=backend_state, config={'paths': {'project_root': '/old/server'}},
                   extra={'optimizer': {'large': torch.ones(128)}}, rng='not for inference')
    restorer = dict(format='rgb_restoration_diffusion_v1',
                    model_config={'architecture': 'pixel_diffusion_nafnet', 'width': 4},
                    model={'stem.weight': torch.arange(12, dtype=torch.float32),
                           'eta': torch.tensor([0., .5, 1.], dtype=torch.float32)},
                    optimizer={'momentum': torch.ones(100)}, rng='discard')
    paths = {}
    for name, payload in (('backend', backend), ('restorer', restorer)):
        paths[name] = tmp_path / (name + '.pt')
        torch.save(payload, paths[name])
    terminal = dict(format='semantic_grain_terminal_only_v1', terminal_only=True,
                    do_not_replace_geometry_upstream=True, complete=True, step=64, total_steps=64,
                    architecture=architecture['semantic_decoder'], semantic_state_dict=head,
                    identity=dict(base_checkpoint_sha256=sha(paths['backend']), initial_head_sha256=state_digest(head)),
                    optimizer={'state': {'large': torch.ones(100)}}, rng='discard', snapshots=[1, 2, 3])
    paths['terminal'] = tmp_path / 'terminal.pt'
    torch.save(terminal, paths['terminal'])
    return paths


def environment():
    return dict(python='3.12.3', cuda='12.8', packages={name: '1.2.3' for name in (*REQUIRED_PACKAGES, 'opencv-python')},
                API_TOKEN='synthetic-must-not-be-copied')


def source_fixture(tmp_path):
    root = tmp_path / 'runtime'
    files = {
        'tools/infer_semifinal.py': 'def main():\n    pass\n',
        'models/example.py': '# synthetic source\n',
        'segment-anything-2/sam2/__init__.py': '# synthetic SAM2 package\n',
        'segment-anything-2/sam2/configs/example.yaml': 'model: synthetic\n',
        'segment-anything-2/LICENSE': 'Synthetic testing license\n',
    }
    rows = []
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf8')
        rows.append(dict(path=name, sha256=sha(path)))
    config = tmp_path / 'template.yaml'
    config.write_text(yaml.safe_dump(dict(paths={'project_root': 'auto'},
                                         sam2={'sam2_repo_path': 'segment-anything-2'}, submission={'models': {}})), encoding='utf8')
    return root, dict(files=rows), config


def complete_code_model(tmp_path, terminal=True):
    paths = checkpoints(tmp_path)
    code = tmp_path / 'submission/code_model'
    build_models(code, paths['backend'], paths['restorer'], paths['terminal'] if terminal else None)
    root, manifest, config = source_fixture(tmp_path)
    build_code(code, root, manifest, config, environment())
    return code


def result_fixture(tmp_path):
    originals, results = tmp_path / 'originals', tmp_path / 'submission/results'
    originals.mkdir()
    results.mkdir(parents=True)
    for name in NAMES:
        cv2.imwrite(str(originals / (name + '.png')), np.zeros((6, 8, 3), dtype=np.uint8))
        ids = np.zeros((6, 8), dtype=np.uint16)
        ids[:3] = 300
        ids[3:] = 65535
        assert cv2.imwrite(str(results / (name + '_inst.png')), ids)
        write_json(results / (name + '_class.json'), {'300': 0, '65535': 1})
    return originals, results


@pytest.mark.parametrize('kind', ['backend', 'restorer', 'terminal_semantic'])
def test_export_preserves_all_tensor_values_and_drops_training_records(tmp_path, kind):
    paths = checkpoints(tmp_path)
    source = paths['terminal' if kind == 'terminal_semantic' else kind]
    destination = tmp_path / ('export_' + kind + '.pt')
    receipt = export_checkpoint(source, destination, kind, base_sha=sha(paths['backend']),
                                base_head_sha=state_digest({'projection.weight': torch.tensor([[1.25, -.375]])}))
    payload = torch.load(destination, weights_only=True)
    assert not any(key in payload for key in ('optimizer', 'rng', 'snapshots', 'config', 'extra'))
    key = dict(backend='model_state_dict', restorer='model', terminal_semantic='head_state_dict')[kind]
    assert state_digest(payload[key]) == receipt['state_sha256']
    assert receipt['sha256'] == sha(destination) and receipt['fp32_exact']
    if kind == 'backend':
        assert payload['architecture']['sam2']['sam2_repo_path'] == 'segment-anything-2'
    if kind == 'terminal_semantic':
        assert payload['format'] == 'semantic_grain_inference_v1' and payload['steps'] == 64


def test_export_rejects_half_precision_instead_of_changing_it(tmp_path):
    paths = checkpoints(tmp_path)
    payload = torch.load(paths['backend'], weights_only=False)
    payload['model_state_dict']['encoder.weight'] = payload['model_state_dict']['encoder.weight'].half()
    torch.save(payload, paths['backend'])
    with pytest.raises(ValueError, match='unchanged FP32'):
        export_checkpoint(paths['backend'], tmp_path / 'bad.pt', 'backend')
    assert not (tmp_path / 'bad.pt').exists()


def test_terminal_original_source_identity_cannot_be_overwritten(tmp_path):
    paths = checkpoints(tmp_path)
    with pytest.raises(ValueError, match='original base/head identity'):
        export_checkpoint(paths['terminal'], tmp_path / 'bad.pt', 'terminal_semantic',
                          base_sha='a' * 64, base_head_sha='b' * 64)


@pytest.mark.parametrize('name', ['../x.py', 'tools/../../x.py', '/x.py', 'a\\b.py',
                                 'C:/x.py', 'tools//x.py', './x.py'])
def test_relative_paths_reject_archive_escape(name):
    with pytest.raises(ValueError, match='Unsafe relative path'):
        safe_relative(name)


def test_manifest_rejects_artifact_source_and_unreal_inference(tmp_path):
    _, original, _ = source_fixture(tmp_path)
    poisoned = deepcopy(original)
    poisoned['files'].append(dict(path='outputs/private.pt', sha256='a' * 64))
    with pytest.raises(ValueError, match='forbidden'):
        source_rows(poisoned)
    missing = deepcopy(original)
    missing['files'] = [row for row in missing['files'] if row['path'] != 'tools/infer_semifinal.py']
    with pytest.raises(ValueError, match='Real standalone'):
        source_rows(missing)


def test_build_complete_code_uses_export_hashes_and_omits_secrets(tmp_path):
    code = complete_code_model(tmp_path)
    manifest = validate_code_model(code)
    config = yaml.safe_load((code / 'config/inference/submission_reflect.yaml').read_text(encoding='utf8'))
    assert config['paths']['project_root'] == 'auto'
    for kind, name in MODEL_NAMES.items():
        assert config['submission']['models'][kind] == dict(file=name, sha256=sha(code / 'model' / name))
    assert 'synthetic-must-not-be-copied' not in (code / 'environment.json').read_text(encoding='utf8')
    assert 'opencv-python==1.2.3' in (code / 'requirements.txt').read_text(encoding='utf8')
    assert 'opencv-contrib-python' not in (code / 'requirements.txt').read_text(encoding='utf8')
    assert manifest['complete'] and all(row['sha256'] == sha(code / row['path']) for row in manifest['files'])


def test_code_manifest_rejects_tampering_or_undeclared_cache(tmp_path):
    code = complete_code_model(tmp_path)
    (code / 'model/backend.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='sealed SHA differs'):
        validate_code_model(code)


def test_exact_100_results_shape_id_and_uint16_contract(tmp_path):
    originals, results = result_fixture(tmp_path)
    rows = validate_results(results, originals)
    assert len(rows) == 100 and rows[0]['source'] == 'test_069' and rows[-1]['source'] == 'test_168'
    assert all(row['instances'] == 2 and row['shape'] == [6, 8] for row in rows)


@pytest.mark.parametrize('bad', ['boolean', 'float', 'missing_id', 'leading_zero', 'duplicate_key', 'array'])
def test_results_reject_noncanonical_json(tmp_path, bad):
    originals, results = result_fixture(tmp_path)
    texts = {'boolean': '{"300": true, "65535": 1}', 'float': '{"300": 0.0, "65535": 1}',
             'missing_id': '{"300": 0}', 'leading_zero': '{"0300": 0, "65535": 1}',
             'duplicate_key': '{"300": 0, "300": 1, "65535": 1}', 'array': '[0, 1]'}
    (results / 'test_069_class.json').write_text(texts[bad], encoding='utf8')
    with pytest.raises(ValueError):
        validate_results(results, originals)


@pytest.mark.parametrize('bad', ['uint8', 'rgb', 'wrong_shape', 'bad_filename', 'extra_copy'])
def test_results_reject_wrong_png_or_cohort(tmp_path, bad):
    originals, results = result_fixture(tmp_path)
    path = results / 'test_069_inst.png'
    if bad == 'uint8':
        cv2.imwrite(str(path), np.ones((6, 8), dtype=np.uint8))
    elif bad == 'rgb':
        cv2.imwrite(str(path), np.ones((6, 8, 3), dtype=np.uint16))
    elif bad == 'wrong_shape':
        cv2.imwrite(str(path), np.ones((5, 8), dtype=np.uint16))
    elif bad == 'bad_filename':
        path.rename(results / 'test_69_inst.png')
    else:
        (results / 'copy.png').write_bytes(path.read_bytes())
    with pytest.raises(ValueError):
        validate_results(results, originals)


def test_complete_zip_has_only_results_and_code_model_and_verified_crc(tmp_path):
    complete_code_model(tmp_path)
    originals, _ = result_fixture(tmp_path)
    archive = tmp_path / '复赛结果_待填写.zip'
    receipt = pack_submission(tmp_path / 'submission', archive, originals, compression_level=6)
    with zipfile.ZipFile(archive) as zipped:
        assert zipped.testzip() is None
        assert {name.split('/')[0] for name in zipped.namelist()} == {'results', 'code_model'}
        assert sum(name.startswith('results/') for name in zipped.namelist()) == 200
        assert 'code_model/infer.py' in zipped.namelist()
        assert 'code_model/model/semantic_s64.pt' in zipped.namelist()
    assert receipt['result_pairs'] == 100 and receipt['zip_crc_ok'] and receipt['all_member_sha_verified']
    assert receipt['archive_sha256'] == sha(archive) and not receipt['upload_performed']
    assert not receipt['size_limits_verified']


def test_placeholder_or_incomplete_engine_cannot_be_packed(tmp_path):
    originals, _ = result_fixture(tmp_path)
    code = tmp_path / 'submission/code_model'
    code.mkdir()
    (code / 'infer.py').write_text('# placeholder', encoding='utf8')
    with pytest.raises(FileNotFoundError):
        pack_submission(tmp_path / 'submission', tmp_path / 'bad.zip', originals)
    assert not (tmp_path / 'bad.zip').exists()


def test_code_model_without_terminal_is_valid_and_smaller_ledger(tmp_path):
    code = complete_code_model(tmp_path, terminal=False)
    validate_code_model(code)
    config = yaml.safe_load((code / 'config/inference/submission_reflect.yaml').read_text(encoding='utf8'))
    assert config['submission']['models']['terminal_semantic'] is None
    assert not (code / 'model/semantic_s64.pt').exists()
    assert json.loads((code / 'inference_export.json').read_text(encoding='utf8'))['parameter_ledger']['conservative_total'] == 84951045


def test_environment_requires_real_exact_versions_and_unambiguous_opencv():
    missing = environment()
    del missing['packages']['torch']
    with pytest.raises(ValueError, match='incomplete'):
        pinned_requirements(missing)
    extra = environment()
    extra['packages']['opencv-contrib-python'] = '1.2.3'
    with pytest.raises(ValueError, match='Exactly one'):
        pinned_requirements(extra)
