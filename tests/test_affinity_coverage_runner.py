# -*- coding: utf-8 -*-
"""训练启动边界、CPU-grid来源拒绝与short gate的完整契约。"""
from copy import deepcopy
import json

import pytest

from tools import run_affinity_coverage as runner
from data.affinity_coverage import REFERENCE_SHA, REFERENCE_STEPS_SHA, TRIAL_SHA, METADATA_SHA, sha
from utils.config import load_config


def test_formal_requires_separate_go_and_smoke_never_continues_or_resumes():
    with pytest.raises(RuntimeError,match='independent explicit'):runner.run(smoke=False)
    with pytest.raises(RuntimeError,match='does not resume'):runner.run(smoke=True,go20=True)
    with pytest.raises(RuntimeError,match='does not resume'):runner.run(smoke=True,resume='untrusted.pt')


def test_actual_grid_gate_protects_cache_identity_before_gpu(tmp_path,monkeypatch):
    data=tmp_path/'cached_map_receipt.json';data.write_text('{}',encoding='utf8')
    report=dict(complete=True,format='affinity_coverage_grid_v1',checkpoint_sha256=REFERENCE_SHA,
        epochs=20,original_updates=1280,original49_sources=49,cached148_sources=148,
        actual_four_draw_grid_content_exact=[24,25,28,37],actual_target_mask_gate=True,
        protected_before_after_exact=True,full_processed_newgt_including_filled=True,
        parents=dict(trial_sha256=TRIAL_SHA,trial_metadata_sha256=METADATA_SHA,original_steps_sha256=REFERENCE_STEPS_SHA),
        protected_sha256={str(data):sha(data)},newgt={'canonical':'testonly'})
    path=tmp_path/'report.json';path.write_text(json.dumps(report),encoding='utf8');monkeypatch.setattr(runner,'GRID',path)
    result=runner.grid_contract({});assert result['protected_sha256']==report['protected_sha256']
    data.write_text('{"changed":true}',encoding='utf8')
    with pytest.raises(RuntimeError,match='Grid protected'):runner.grid_contract({})


@pytest.mark.parametrize('mutation',['candidate_contract','budget','running','configuration','control_retrain'])
def test_formal_gate_rejects_incomplete_or_hidden_changes(tmp_path,monkeypatch,mutation):
    cfg=load_config(runner.CONFIG_PATH);contract={'source_sha':'fixed','newgt':'current','grid':'complete'};audit={'passed':True}
    parity={'passed':True}
    state=dict(status='complete',stage='done',smoke=True,completed=['parity','training'],contract=deepcopy(contract),
        config=runner.canonical(cfg),control_retrained=False,audit=audit,parity=parity)
    monkeypatch.setattr(runner,'SMOKE',tmp_path)
    monkeypatch.setattr(runner,'verify_snapshot',lambda *args,**kwargs:None)
    monkeypatch.setattr(runner,'verify_run',lambda *args,**kwargs:audit)
    (tmp_path/'parity.json').write_text(json.dumps(parity),encoding='utf8')
    contract['sources']={};state['contract']['sources']={}
    (tmp_path/'pipeline_status.json').write_text(json.dumps(state),encoding='utf8')
    assert runner.verify_gate(cfg,contract)['status']=='complete'
    if mutation=='candidate_contract':state['contract']['grid']='different'
    elif mutation=='budget':state['completed']=['parity']
    elif mutation=='running':state['status']='running'
    elif mutation=='configuration':state['config']['backend_adaptation']['pseudo_weight']=1.
    else:state['control_retrained']=True
    (tmp_path/'pipeline_status.json').write_text(json.dumps(state),encoding='utf8')
    with pytest.raises(RuntimeError):runner.verify_gate(cfg,contract)
