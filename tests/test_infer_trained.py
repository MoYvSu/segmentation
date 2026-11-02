# -*- coding: utf-8 -*-
"""真实verify_export清单生产端对接、身份拒绝、零噪C适配和路由；不运行GPU。"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import infer as router
from repro import pipeline
from tools import infer_trained as trained


@pytest.fixture
def exported(tmp_path, monkeypatch):
    monkeypatch.setattr(trained, 'ROOT', tmp_path)
    run = tmp_path / 'outputs/training_run'
    files = dict(backend=run/'export_backend/model/backend.pt',
                 restorer=run/'export_backend/model/restorer.pt', naf_export=run/'naf_export/model/naf.pt')
    naf_state = {'weight':torch.tensor([.1,.2,.3,.4])}
    boundary_sha = trained.naf.state_sha256(SimpleNamespace(state_dict=lambda:naf_state))
    payloads = dict(
        backend=dict(format=trained.MODEL_FORMATS['backend'], architecture={'sam2':{}},
                     model_state_dict={'weight':torch.tensor([1.,2.])}),
        restorer=dict(format=trained.MODEL_FORMATS['restorer'],
                      model_config=dict(architecture='pixel_diffusion_nafnet',num_steps=16,kappa=.03),
                      model={'weight':torch.tensor([3.,4.,5.])}),
        naf_export=dict(format=trained.MODEL_FORMATS['naf_export'], model_state_dict=naf_state,
            global_step=1000, model_sha256=boundary_sha, input_contract=deepcopy(trained.naf.INPUT_CONTRACT),
            identity=dict(numerical_contract=deepcopy(trained.naf.NUMERICAL_CONTRACT))))
    for role, file in files.items():
        file.parent.mkdir(parents=True,exist_ok=True)
        torch.save(payloads[role],file)
    context = dict(run_dir=str(run), artifacts={key:str(path) for key,path in files.items()})
    # 使用真实生产函数，而不是手工模仿其manifest字段。
    pipeline.verify_export(context)
    manifest = run/'verify_export/manifest.json'
    config = trained.entry.ROOT/'config/inference/area_floor_200.yaml'
    args = SimpleNamespace(training_manifest=str(manifest), input_dir=str(tmp_path/'images'),
        output_dir=str(tmp_path/'outputs/prediction'), config=str(config), dataset=None, only=['test_169'])
    return SimpleNamespace(root=tmp_path,run=run,files=files,payloads=payloads,manifest=manifest,
                           args=args,naf_sha=boundary_sha)


def save_manifest(h, document):
    h.manifest.write_text(json.dumps(document),encoding='utf-8')


def test_real_pipeline_export_manifest_separate_directories_and_state_algorithms(exported):
    h = exported
    document, identity, files, run = trained.validate_manifest(h.manifest)
    assert document['format'] == 'current_training_export_manifest_v1'
    assert run == h.run and files == h.files
    assert identity['state_elements_parameter_upper_bound'] == 9
    assert identity['models']['boundary_naf']['state_sha256'] == h.naf_sha
    assert identity['models']['boundary_naf']['entry_state_sha256'] != h.naf_sha
    assert identity['original_model_identity_claimed'] is False


@pytest.mark.parametrize('defect', ['file_bytes','budget','count','format','outside_run','NAF_state','FP16','nonfinite'])
def test_invalid_actual_manifest_or_checkpoint_rejected_before_inference(exported, defect):
    h = exported
    document = json.loads(h.manifest.read_text())
    if defect == 'file_bytes': h.files['backend'].write_bytes(b'changed')
    elif defect == 'budget': document['state_elements_parameter_upper_bound'] = 500_000_000
    elif defect == 'count': document['models']['backend']['state_tensors'] += 1
    elif defect == 'format': document['models']['backend']['format'] = 'unknown'
    elif defect == 'outside_run':
        other = h.root/'outside.pt'; other.write_bytes(h.files['backend'].read_bytes())
        document['models']['backend']['path'] = str(other)
    else:
        role = 'naf_export' if defect == 'NAF_state' else 'backend'
        payload = deepcopy(h.payloads[role])
        if defect == 'NAF_state': payload['model_sha256'] = '0'*64
        elif defect == 'FP16': payload['model_state_dict']['weight'] = payload['model_state_dict']['weight'].half()
        else: payload['model_state_dict']['weight'][0] = float('nan')
        torch.save(payload,h.files[role])
        document['models'][role]['file_sha256'] = trained.entry.sha(h.files[role])
    save_manifest(h,document)
    with pytest.raises(ValueError): trained.validate_manifest(h.manifest)


def scoped_targets():
    return [(trained.naf,'EXPECTED_MODEL_SHA'),(trained.flat,'BACKEND_SHA'),(trained.flat,'RESTORER_SHA'),
        *[(trained.entry,name) for name in ('load_config','model_paths','validate_config',
            'configure_restoration_noise','runtime_record','write')]]


@pytest.mark.parametrize('fail', [None,'runtime','identity'])
def test_new_route_uses_actual_export_paths_zero_audit_and_restores_old_identity(exported, monkeypatch, fail):
    h = exported
    monkeypatch.setattr(trained.entry,'runtime_record',lambda **kw:dict(precision='FP32',restoration_noise=kw))
    targets = scoped_targets()
    originals = [getattr(owner,name) for owner,name in targets]

    def compact(args, *, device):
        assert device == 'cpu' and args.geometry_mode == 'C' and args.global_noise == 'zero'
        assert Path(args.model_dir) == h.run and Path(args.naf_checkpoint) == h.files['naf_export']
        assert args.expected_model_sha == h.naf_sha
        assert trained.naf._s0_expected_model_sha(args.expected_model_sha) == h.naf_sha
        cfg = trained.entry.load_config(args.config)
        trained.flat.execution_config(cfg)  # 真实旧C/zero/final200校验，在显式身份scope内通过。
        paths = trained.entry.model_paths(cfg,Path(args.model_dir))
        assert paths == {key:h.files[key] for key in ('backend','restorer')}
        assert all(str(path.relative_to(Path(args.model_dir))) for path in paths.values())
        restorer = SimpleNamespace()
        trained.entry.configure_restoration_noise(restorer,trained.zero.ZERO_MODE)
        assert restorer.inference_noise_mode == trained.zero.ZERO_MODE
        report = dict(complete=True,images=[dict(source='test_169')], config=cfg,
            compact_readout=trained.compact.contract('C'),
            runtime=trained.entry.runtime_record(restoration_noise_mode=trained.zero.ZERO_MODE))
        audit = Path(args.output_dir)/'audit'; audit.mkdir(parents=True)
        trained.entry.write(audit/'report.json',report)
        if fail == 'runtime': raise RuntimeError('deliberate inference failure')
        if fail == 'identity': h.files['backend'].write_bytes(b'changed after prediction')
        return report

    monkeypatch.setattr(trained.compact,'run',compact)
    if fail:
        with pytest.raises((RuntimeError,ValueError),match='inference failure|file SHA differs'):
            trained.run(h.args,device='cpu')
        failed = json.loads((Path(h.args.output_dir)/'audit/trained_failed.json').read_text())
        assert failed['complete'] is False and failed['format'] == trained.FORMAT
        saved = json.loads((Path(h.args.output_dir)/'audit/report.json').read_text())
        assert saved['complete'] is False
    else:
        result = trained.run(h.args,device='cpu')
        saved = json.loads((Path(h.args.output_dir)/'audit/trained_report.json').read_text())
        assert result == saved and saved['complete'] is True
        assert 'original_backend_sha256' not in saved['compact_readout']
    assert all(getattr(owner,name) is old for (owner,name),old in zip(targets,originals))
    assert saved['format'] == trained.FORMAT and saved['original_model_identity_claimed'] is False
    assert saved['runtime']['restoration_noise']['name'] == trained.zero.ZERO_MODE
    assert saved['runtime']['restoration_noise']['random_noise_generated'] is False
    assert saved['model_identity']['models']['boundary_naf']['state_sha256'] == h.naf_sha


def test_old_default_still_rejects_new_weights_and_source_output_overlap(exported):
    h = exported
    base = trained.entry.load_config(h.args.config)
    base['submission']['models']['backend']['sha256'] = '1'*64
    with pytest.raises(ValueError,match='original 13b3'): trained.flat.execution_config(base)
    for output in (h.run/'predictions',Path(h.args.input_dir)/'predictions',h.root/'public_prediction'):
        with pytest.raises(ValueError): trained.safe_output(output,h.args.input_dir,h.run)


def test_frontend_default_and_explicit_source_dispatch_and_engine_conflict(monkeypatch):
    from tools import infer_compact
    observed = []
    monkeypatch.setattr(infer_compact,'parse_args',lambda argv:observed.append(('old',list(argv))) or SimpleNamespace())
    monkeypatch.setattr(infer_compact,'run',lambda args:dict(complete=True,geometry_mode='C',images=[],actual_compact_counts={}))
    router.main(['--input-dir','images','--output-dir','out','--naf-checkpoint','old.pt'])
    assert observed[-1][0] == 'old'
    monkeypatch.setattr(trained,'main',lambda argv:observed.append(('trained',list(argv))))
    router.main(['--training-manifest','run/verify_export/manifest.json','--input-dir','i','--output-dir','o'])
    assert observed[-1] == ('trained',['--input-dir','i','--output-dir','o','--training-manifest','run/verify_export/manifest.json'])
    before = len(observed)
    with pytest.raises(SystemExit): router.main(['--backend','tensorrt','--training-manifest','x'])
    assert len(observed) == before
    with pytest.raises(SystemExit):
        trained.parse_args(['--training-manifest','x','--input-dir','i','--output-dir','o','--model-dir','other'])
