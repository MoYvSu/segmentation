# -*- coding: utf-8 -*-
"""关系头的显式启用、C终态衔接和分数继承；只用合成数据与CPU。"""
from argparse import Namespace
import builtins
from copy import deepcopy
import json
import sys

import numpy as np
import pytest

from tools import infer_boundary_naf as wrapper
from utils import boundary_relation_inference as relation_module
from test_boundary_relation_inference import checkpoint, scene
from test_infer_boundary_naf import install_fake_entry, one_cpu_thread
from test_infer_boundary_naf_support import install_support


def options(**values):
    return Namespace(model_dir=None, **values)


def test_disabled_resolver_has_no_changes_or_extra_sources():
    cfg={'boundary_relation':{'enabled':False},'inference':{'sentinel':[1,2]}}
    before=deepcopy(cfg);args=options();old_args=deepcopy(vars(args))
    assert wrapper._relation_options(args,cfg) is None
    assert cfg==before and vars(args)==old_args
    base=wrapper._sources()
    assert base==wrapper._sources(include_relation=False)
    assert all('boundary_relation' not in name for name in base)
    enabled=wrapper._sources(include_relation=True)
    assert {name:enabled[name] for name in base}==base
    assert set(enabled)-set(base)=={'models/boundary_relation.py',
        'utils/boundary_relation_inference.py','utils/boundary_relation_merge.py',
        'utils/boundary_relation_supervision.py'}


@pytest.mark.parametrize('name,value',[
    ('relation_threshold',0),('relation_batch_size',64),('relation_expected_model_sha','a'*64)])
def test_disabled_resolver_rejects_unused_cli_settings(name,value):
    with pytest.raises(ValueError,match='未启用关系头'):
        wrapper._relation_options(options(**{name:value}),{})


def test_configured_basename_resolves_under_relative_model_dir(tmp_path,monkeypatch):
    monkeypatch.chdir(tmp_path)
    args=options();args.model_dir='models'
    cfg={'boundary_relation':{'enabled':True,'checkpoint_file':'relation.pt',
        'threshold':.1,'batch_size':32,'expected_model_sha':'a'*64}}
    before=deepcopy(cfg)
    resolved=wrapper._relation_options(args,cfg)
    assert resolved==dict(checkpoint=tmp_path/'models/relation.pt',threshold=.1,batch_size=32,
        expected_model_sha='a'*64)
    assert cfg==before
    args.model_dir=None
    assert wrapper._relation_options(args,cfg)['checkpoint']==wrapper.ROOT/'model/relation.pt'


@pytest.mark.parametrize('filename',[None,'','../relation.pt','nested/relation.pt',17])
def test_configured_checkpoint_must_be_single_basename(filename):
    with pytest.raises(ValueError,match='单个文件名'):
        wrapper._relation_options(options(),{'boundary_relation':{
            'enabled':True,'checkpoint_file':filename}})


def test_explicit_cli_checkpoint_and_settings_override_config(tmp_path):
    args=options(relation_checkpoint=str(tmp_path/'chosen.pt'),relation_threshold=.1,
        relation_batch_size=64,relation_expected_model_sha='b'*64)
    cfg={'boundary_relation':{'enabled':False,'checkpoint_file':'ignored.pt',
        'threshold':.2,'batch_size':32,'expected_model_sha':'a'*64}}
    actual=wrapper._relation_options(args,cfg)
    assert actual==dict(checkpoint=str(tmp_path/'chosen.pt'),threshold=.1,batch_size=64,
        expected_model_sha='b'*64)


def make_result(data):
    return dict(instances=data[3],classes=data[4],boundary=data[1],
        phases=wrapper.entry.phase_description(data[3],data[4]),
        instance_scores={'1':.91,'2':.31},sentinel={'original':[1,2]})


def assert_old_result_unchanged(result,before):
    assert result.keys()==before.keys()
    for key in before:
        if isinstance(before[key],np.ndarray):assert np.array_equal(result[key],before[key])
        else:assert result[key]==before[key]


def test_apply_zero_threshold_real_processor_exact_and_preserves_old_result():
    data=list(scene(2));data[4]={'1':1,'2':1}
    result=make_result(data);before=deepcopy(result)
    processor=relation_module.BoundaryRelationPostprocessor(None,threshold=0,device='cpu')
    output,audit=wrapper._apply_relation_result(result,*data[:3],processor)
    assert_old_result_unchanged(result,before)
    assert np.array_equal(output['instances'],result['instances']) and output['classes']==result['classes']
    assert output['instance_scores']==result['instance_scores']
    assert output is not result and output['instances'] is not result['instances']
    assert output['classes'] is not result['classes']
    assert audit['no_op_reason']=='threshold_zero' and audit['loaded_parameter_count']==0


def test_apply_actual_merge_inherits_survivor_score_without_revote(tmp_path,monkeypatch):
    data=list(scene(2));data[4]={'1':1,'2':1}
    result=make_result(data);before=deepcopy(result)
    path,payload=checkpoint(tmp_path)
    processor=relation_module.BoundaryRelationPostprocessor(path,device='cpu',
        expected_model_sha=payload['model_sha256'])
    def forbidden(*args,**kwargs):pytest.fail('合并以后不得再投票')
    monkeypatch.setattr(wrapper.entry.views,'revote_instances',forbidden)
    output,audit=wrapper._apply_relation_result(result,*data[:3],processor)
    assert_old_result_unchanged(result,before)
    assert set(np.unique(output['instances']))=={0,1}
    assert output['classes']=={'1':1} and output['instance_scores']=={'1':.91}
    assert output['instance_score_policy']=='relation_survivor_original_score_inherited_not_recomputed'
    assert output['phases']==wrapper.entry.phase_description(output['instances'],output['classes'])
    assert audit['merge']['selected_count']==1 and audit['merge']['filled_seam_pixels']>0
    assert audit['input_unchanged']


def test_disabled_wrapper_never_imports_or_constructs_relation_and_preserves_old_count(tmp_path,monkeypatch):
    args,state,output,naf,base_count,*_=install_fake_entry(tmp_path,monkeypatch)
    original_import=builtins.__import__
    def guarded_import(name,*args,**kwargs):
        if name in ('models.boundary_relation','utils.boundary_relation_inference',
                    'utils.boundary_relation_merge','utils.boundary_relation_supervision'):
            pytest.fail('默认关闭时不能导入关系模块')
        return original_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded_import)
    def forbidden(*args,**kwargs):pytest.fail('默认关闭时不能构造关系头')
    monkeypatch.setattr(relation_module,'BoundaryRelationPostprocessor',forbidden)
    relation_modules={k for k in sys.modules if 'boundary_relation' in k}
    receipt=wrapper.run(args,device='cpu')
    assert {k for k in sys.modules if 'boundary_relation' in k}==relation_modules
    assert receipt['parameter_count']==base_count+sum(p.numel() for p in naf.parameters())
    assert 'boundary_relation' not in receipt and not (output/'audit/relation').exists()
    report=json.loads((output/'audit/report.json').read_text(encoding='utf8'))
    assert 'boundary_relation' not in report['models'] and not report.get('parameter_count_includes_relation',False)
    assert all('boundary_relation' not in name for name in receipt['imported_sources'])
    assert state['raw_predict_calls']==1


@pytest.mark.parametrize('seed_support,config_name',[
    ('none','area_floor_200.yaml'),('original-zero','area_floor_100.yaml')])
def test_relation_rejects_non_C_terminal_policy_before_models_or_prediction(tmp_path,monkeypatch,seed_support,config_name):
    args,state,output,*_=install_fake_entry(tmp_path,monkeypatch)
    args.seed_support=seed_support;args.config=str(wrapper.ROOT/'config/inference'/config_name)
    args.relation_checkpoint=str(tmp_path/'relation.pt')
    with pytest.raises(ValueError,match='C终态'):
        wrapper.run(args,device='cpu')
    assert state['helper_calls']==state['raw_predict_calls']==0 and not output.exists()


def test_relation_rejects_other_NAF_state_even_with_correct_S0_and_final200(tmp_path,monkeypatch):
    args,state,output,*_=install_fake_entry(tmp_path,monkeypatch)
    args.seed_support='original-zero';args.config=str(wrapper.ROOT/'config/inference/area_floor_200.yaml')
    args.relation_checkpoint=str(tmp_path/'relation.pt')
    # 合法toy NAF的真实SHA不同于20a59，不能因架构相同就混入部署。
    with pytest.raises(ValueError,match='20a59'):
        wrapper.run(args,device='cpu')
    assert state['helper_calls']==state['raw_predict_calls']==0 and not output.exists()


@pytest.mark.parametrize('parameter_budget_fail',[False,True])
def test_enabled_wrapper_applies_only_after_original_predict_and_writes_private_audit(tmp_path,monkeypatch,parameter_budget_fail):
    args,state,output,naf,base_count,*_=install_fake_entry(tmp_path,monkeypatch)
    args.config=str(wrapper.ROOT/'config/inference/area_floor_200.yaml')
    args.seed_support='original-zero';args.relation_checkpoint=str(tmp_path/'relation.pt')
    args.relation_expected_model_sha='b'*64
    supported,_=install_support(monkeypatch,state)
    original_load=wrapper.load_naf_checkpoint
    def synthetic_naf_bridge(*args,**kwargs):
        model,meta=original_load(*args,**kwargs)
        # 只用toy NAF测试入口顺序；生产loader的状态校验由旧测试独立覆盖。
        return model,dict(meta,model_sha256='20a59c3971b2112c48a3b744c31f8051f5687772456cd35b5d38f0ae41b158e8')
    monkeypatch.setattr(wrapper,'load_naf_checkpoint',synthetic_naf_bridge)
    events=[];original_predict=wrapper.entry.predict
    def original_finishes(*args,**kwargs):
        rgb,result=original_predict(*args,**kwargs)
        result['instance_scores']={'1':.7}
        state['old_result']=result;state['old_result_copy']=deepcopy(result)
        events.append('original_predict_returned')
        return rgb,result
    monkeypatch.setattr(wrapper.entry,'predict',original_finishes)
    class FakeRelation:
        parameter_count=211865
        def __init__(self,**kwargs):
            self.threshold=kwargs['threshold'];self.calls=0;self.verified=0
            assert kwargs['expected_model_sha']=='b'*64 and kwargs['device']=='cpu'
            state['relation']=self
        def process(self,rgb,before,after,ids,classes):
            assert events==['original_predict_returned']
            assert np.array_equal(before,state['before'][0,0].numpy()) and np.all(after==np.float32(.8))
            assert ids is state['old_result']['instances'] and classes is state['old_result']['classes']
            self.calls+=1;events.append('relation')
            return ids.copy(),classes.copy(),dict(pairs=[],private_pair_diagnostic='audit-only',
                merge=dict(selected_count=0,filled_seam_pixels=0))
        def provenance(self):return dict(mode='rgb',model_sha256='b'*64,parameter_count=self.parameter_count)
        def verify_unchanged(self):self.verified+=1;return True
    monkeypatch.setattr(relation_module,'BoundaryRelationPostprocessor',FakeRelation)
    expected=base_count+sum(p.numel() for p in naf.parameters())+211865
    if parameter_budget_fail:
        monkeypatch.setattr(wrapper,'MAX_PARAMETERS',expected)
        with pytest.raises(RuntimeError,match='parameters exceed'):
            wrapper.run(args,device='cpu')
        assert state['raw_predict_calls']==state['relation'].calls==0 and events==[]
        return
    receipt=wrapper.run(args,device='cpu')
    assert events==['original_predict_returned','relation'] and state['relation'].calls==1
    assert state['relation'].verified==1 and supported['replay']==1
    assert_old_result_unchanged(state['old_result'],state['old_result_copy'])
    assert receipt['parameter_count']==expected and receipt['boundary_relation']['parameter_count']==211865
    report=json.loads((output/'audit/report.json').read_text(encoding='utf8'))
    assert report['parameter_count']==expected and report['parameter_count_includes_relation']
    assert report['models']['boundary_relation']['parameter_count']==211865
    record=receipt['images'][0]['boundary_relation']
    assert record['audit']=='audit/relation/test_169.json'
    pair_audit=json.loads((output/record['audit']).read_text(encoding='utf8'))
    assert pair_audit['private_pair_diagnostic']=='audit-only'
    assert record['audit_sha256']==wrapper.entry.sha(output/record['audit'])
    assert not list((output/'results').iterdir())
    assert not (output/'results/relation_report.json').exists()
    relation_report=json.loads((output/'audit/relation_report.json').read_text(encoding='utf8'))
    assert relation_report['stage']=='after_raw_vote_and_final200'
    assert relation_report['no_revote'] and relation_report['no_second_area_cleanup'] and relation_report['no_WS_rerun']
