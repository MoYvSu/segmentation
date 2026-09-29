# -*- coding: utf-8 -*-
"""短测JSON回执的表示变化不能误拦正式启动，真实预算变化仍须拒绝。"""
import json
from copy import deepcopy
from pathlib import Path
import hashlib

import pytest

from tools import run_semantic_consistency as runner
from tools.run_semantic_consistency import same_config, control_config, validate_reused_control
from utils.config import load_config


def test_json_receipt_preserves_resolved_configuration_identity():
    config = load_config('config/train/semantic_consistency.yaml')
    receipt = json.loads(json.dumps(config))
    assert same_config(config, receipt)


def test_budget_change_cannot_reuse_an_old_short_test():
    config = load_config('config/train/semantic_consistency.yaml')
    receipt = json.loads(json.dumps(config))
    receipt['semantic_adaptation']['epochs'] = 60
    assert not same_config(config, receipt)


def reused_fixture(tmp_path):
    config = load_config('config/train/semantic_gray3.yaml')
    old = load_config('config/train/semantic_gray.yaml')
    for cfg in (config,old):
        cfg['paths']['project_root'] = str(tmp_path)
    old_dir, short_dir = tmp_path/'old', tmp_path/'short'
    old_dir.mkdir();short_dir.mkdir()
    images = Path(tmp_path/config['inference']['test_dir']);images.mkdir(parents=True)
    (images/'test_001.jpg').write_bytes(b'file-list-only')
    checkpoint = old_dir/'epoch_020.pt';checkpoint.write_bytes(b'unchanged-checkpoint-fixture')
    flags = {k:True for k in runner.COMPLETION_FLAGS}
    common = dict(flags, arm='control',status='completed',failed_updates=0,
                  sources={'original':'fixed'},data={'sources':32},frozen_state_sha256='frozen',
                  d5a_state_sha256='d5a',initial_semantic_sha256='semantic')
    final = dict(common,config=old,epoch=20,updates=1280,smoke=False,final_checkpoint=str(checkpoint))
    short = dict(common,config=config,epoch=1,updates=6,smoke=True)
    (old_dir/'status.json').write_text(json.dumps(final))
    (short_dir/'status.json').write_text(json.dumps(short))
    steps=[]
    for index in range(1280):
        row={k:index for k in runner.STREAM_FIELDS}
        row.update(semantic_loss=.05,unlabeled=None)
        steps.append(row)
    (old_dir/'steps.jsonl').write_text('\n'.join(json.dumps(s) for s in steps))
    (short_dir/'steps.jsonl').write_text('\n'.join(json.dumps(s) for s in steps[:6]))
    deployment=old_dir/'deployment';deployment.mkdir()
    restore=config['semantic_adaptation']['restoration']
    manifest=dict(epoch=20,config=old,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  d5a_sha256=restore['checkpoint_sha256'],sampling_mode=restore['mode'],precision='FP32',
                  images=[{'image':'test_001.jpg'}])
    (deployment/'manifest.json').write_text(json.dumps(manifest))
    for suffix in ('_inst.png','_class.json','_class_confidence.json'):
        (deployment/('test_001'+suffix)).write_bytes(b'existence-only')
    return config,old_dir,short_dir


def test_reuse_accepts_candidate_only_changes_and_replayed_supervised_prefix(tmp_path):
    cfg,old,short=reused_fixture(tmp_path)
    result=validate_reused_control(old,cfg,short)
    assert result['passed'] and result['prefix_replayed_steps']==6
    assert result['checkpoint']==str(old/'epoch_020.pt')


@pytest.mark.parametrize('change', ['seed','learning_rate','augmentation','initial_weights','inference'])
def test_reuse_rejects_changes_affecting_control(tmp_path,change):
    cfg,old,short=reused_fixture(tmp_path)
    if change=='seed':cfg['semantic_adaptation']['seed']+=1
    elif change=='learning_rate':cfg['semantic_adaptation']['learning_rates']['simple']*=2
    elif change=='augmentation':cfg['semantic_consistency']['appearance']['probability']=.2
    elif change=='initial_weights':cfg['semantic_consistency']['initial_sha256']='different'
    else:cfg['inference']['boundary_threshold']=.1
    with pytest.raises(RuntimeError,match='configuration differs'):
        validate_reused_control(old,cfg,short)


@pytest.mark.parametrize('failure', ['prefix_loss','prefix_input','checkpoint','missing_prediction','data'])
def test_reuse_rejects_changed_evidence_or_incomplete_artifacts(tmp_path,failure):
    cfg,old,short=reused_fixture(tmp_path)
    if failure.startswith('prefix'):
        path=short/'steps.jsonl';lines=[json.loads(s) for s in path.read_text().splitlines()]
        lines[1]['semantic_loss' if failure=='prefix_loss' else 'input_sha256']='changed'
        path.write_text('\n'.join(json.dumps(s) for s in lines))
    elif failure=='checkpoint':(old/'epoch_020.pt').write_bytes(b'changed')
    elif failure=='missing_prediction':(old/'deployment/test_001_class.json').unlink()
    else:
        path=short/'status.json';receipt=json.loads(path.read_text());receipt['data']={'sources':31};path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError):validate_reused_control(old,cfg,short)


def test_formal_pipeline_reuses_control_without_training_or_inferring_it(tmp_path,monkeypatch):
    config=load_config('config/train/semantic_gray3.yaml')
    config['paths']['project_root']=str(tmp_path)
    output=tmp_path/config['semantic_adaptation']['output_dir']
    gate=output.with_name(output.name+'_smoke');gate.mkdir(parents=True)
    (gate/'pipeline_status.json').write_text(json.dumps({'status':'completed','config':config}))
    old=tmp_path/config['semantic_consistency']['reuse_control_dir']
    monkeypatch.setattr(runner,'load_config',lambda path:deepcopy(config))
    monkeypatch.setattr(runner,'snapshot_sources',lambda path:None)
    monkeypatch.setattr(runner,'validate_reused_control',lambda *args:{'passed':True,'directory':str(old)})
    pairs=[]
    def pair(*args,**kwargs):pairs.append((args,kwargs));return {'passed':True}
    monkeypatch.setattr(runner,'verify_pair',pair)
    commands=[]
    monkeypatch.setattr(runner.subprocess,'run',lambda command,**kwargs:commands.append(command))
    runner.run_pipeline('already-loaded.yaml')
    assert [c[2] for c in commands]==['train_semantic_consistency.py','train_semantic_d5a.py','tools/render_backend_ablation.py']
    train=commands[0]
    assert train[train.index('--arm')+1]=='prior'
    assert pairs[0][0][0]==old and pairs[0][1]['reuse_control']
    assert 'Control='+str(old/'deployment') in commands[-1]
    assert (output/'control_reference.json').exists() and not (output/'control').exists()
    status=json.loads((output/'pipeline_status.json').read_text())
    assert status['completed']==['train_prior','infer_prior','render']
