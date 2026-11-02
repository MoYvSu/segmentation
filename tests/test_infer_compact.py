# -*- coding: utf-8 -*-
"""compact入口的分支接线、plain patch、raw/final200次数与有效审计。"""
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from tools import infer_compact as infer


def test_plain_patch_does_not_retain_call_history():
    target=SimpleNamespace(fn=lambda value:value)
    replacement=lambda value:value+1
    with infer.PlainPatch.object(target,'fn',side_effect=replacement):
        assert target.fn is replacement and not isinstance(target.fn,Mock)
        assert target.fn(1)==2 and not hasattr(target.fn,'call_args_list')


@pytest.mark.parametrize('mode',['C','F0'])
def test_actual_readout_route_one_raw_vote_one_final200_and_no_extra_F0_stages(monkeypatch,tmp_path,mode):
    cfg=infer.entry.load_config(str(infer.ROOT/'config/inference/area_floor_200.yaml'))
    output=tmp_path/mode
    after=np.full((64,80),.1,np.float32);after[:,39:42]=.95
    semantic=torch.linspace(-3,3,after.size).reshape(1,1,*after.shape)
    if mode=='F0':semantic.zero_()
    raw=np.linspace(.1,.9,after.size,dtype=np.float32).reshape(after.shape)
    def forbidden(*a,**kw):raise AssertionError('Old F0 extra stage reconstruction must be bypassed')
    monkeypatch.setattr(infer.flat,'expected_WS_image',forbidden)
    def run_branch(args,*,device):
        actual_cfg=infer.zero.execution_config(cfg)
        if mode=='C':
            (ids,_),audit=infer.naf._supported_readout(semantic,after,after,actual_cfg)
        else:
            (ids,_),audit=infer.flat.flat_supported_readout(semantic,after,after,actual_cfg,
                                                        original_readout=infer.naf._supported_readout)
            assert audit['flat_semantic_proof']['semantic_base_uint8']==128
        classes,_=infer.entry.views.revote_instances(ids,raw)
        result=infer.entry.finalize_instance_area(dict(instances=ids,classes=classes),actual_cfg)
        infer.entry.verify_prediction(result['instances'],result['classes'],after.shape)
        document=dict(complete=True,format='old',images=[dict(source='synthetic',
            semantic_sha256=infer.compact.array_sha(semantic.numpy()),
            input_boundary_sha256=infer.compact.array_sha(after),output_boundary_sha256=infer.compact.array_sha(after),
            seed_support=audit)])
        (output/'audit').mkdir(parents=True)
        infer.entry.write(output/'audit/report.json',document)
        return document
    monkeypatch.setattr(infer.zero,'run',run_branch)
    monkeypatch.setattr(infer.flat,'run',run_branch)
    args=SimpleNamespace(geometry_mode=mode,config=str(infer.ROOT/'config/inference/area_floor_200.yaml'),
                         output_dir=str(output))
    receipt=infer.run(args,device='cpu')
    assert receipt['format']==infer.FORMAT and receipt['geometry_mode']==mode
    assert receipt['compact_readout_images'][0]['source']=='synthetic'
    assert receipt['actual_compact_counts']==dict(marker_stage_calls=1,reflect_marker_calls=1,
        S0_filter_calls=1,actual_WS_calls=1,temporary_semantic_vote_calls=0,raw_revote_calls=1,final200_calls=1)
    saved=json.loads((output/'audit/compact_report.json').read_text())
    assert saved['compact_readout']['default_geometry']=='C'
    assert saved['compact_readout']['F0_is_separate_candidate'] is True


def test_C_default_and_no_trained_semantic_delta(tmp_path):
    args=infer.parse_args(['--input-dir','i','--output-dir','o','--naf-checkpoint','n'])
    assert args.geometry_mode=='C'
    args.semantic_checkpoint='trained.pt'
    with pytest.raises(ValueError,match='forbidden'):infer.run(args,device='cpu')
