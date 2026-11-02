# -*- coding: utf-8 -*-
"""仅本地CPU对照，不进入交付代码，不加载GPU或真实引擎。"""
import ast
from copy import deepcopy
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'submission/AIC-2026-59388388-代码与数据'
ALIAS = 'standalone_final_inference'
spec = importlib.util.spec_from_file_location(ALIAS, PACKAGE/'inference/__init__.py',
                                             submodule_search_locations=[str(PACKAGE/'inference')])
module = importlib.util.module_from_spec(spec)
sys.modules[ALIAS] = module
spec.loader.exec_module(module)
pipe = importlib.import_module(ALIAS+'.pipeline')
post = importlib.import_module(ALIAS+'.postprocess')

from tools import infer_compact
from tools import semantic_crossover
from tools.boundary_rgb_monitor import predict_boundary_rgb_tiled
from tools.probe_affinity_patch import tensor_from_rgb
from utils import compact_instance_readout, fast_instance_readout, final_instance_cleanup
from utils.affinity_deployment import crop_letterbox_output
from utils.affinity_fusion import affinity_boundary_probability


@pytest.fixture(autouse=True)
def threads():
    old_torch, old_cv = torch.get_num_threads(), cv2.getNumThreads()
    torch.set_num_threads(4); cv2.setNumThreads(2)
    yield
    torch.set_num_threads(old_torch); cv2.setNumThreads(old_cv)


def test_delivery_isolation_has_no_repo_model_or_experiment_imports():
    for path in (PACKAGE/'inference').glob('*.py'):
        source = path.read_text(encoding='utf-8')
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split('.')[0] not in {'tools','utils','models','data','repro'}
        assert 'torch.load(' not in source and 'nn.Module' not in source
        assert 'hashlib' not in source and 'test_169' not in source


@pytest.mark.parametrize('pattern',[0,1,2])
def test_real_WS_raw_vote_final200_exact_old_C_without_audits(pattern):
    config = infer_compact.entry.load_config(str(infer_compact.ROOT/'config/inference/area_floor_200.yaml'))
    after = np.full((64,80),.1,np.float32)
    after[:,39:42] = .95
    if pattern: after[30:33,:] = .9
    if pattern > 1: after[8:16,8:16] = .8
    semantic = torch.linspace(-3,3,after.size).reshape(1,1,*after.shape)
    raw = np.linspace(.13,.87,after.size,dtype=np.float32).reshape(after.shape)
    before = (semantic.clone(),after.copy(),raw.copy())
    with patch.object(compact_instance_readout,'finish_watershed',new=fast_instance_readout.finish_watershed):
        (expected,_),_ = compact_instance_readout.compact_supported_readout(semantic,after,after,config)
    actual = post.watershed_instances(semantic,after,after)
    assert np.array_equal(actual,expected)
    expected_classes,_ = semantic_crossover.revote_instances(expected,raw)
    actual_classes = post.vote_instances(actual,raw)
    assert actual_classes == expected_classes
    expected,expected_classes,_ = final_instance_cleanup.merge_final_instances(expected,expected_classes,200)
    actual,actual_classes = post.merge_final_instances(actual,actual_classes)
    assert np.array_equal(actual,expected) and actual_classes == expected_classes
    assert torch.equal(semantic,before[0]) and np.array_equal(after,before[1]) and np.array_equal(raw,before[2])


def test_probability_mean_strict_half_and_class_inheritance():
    ids = np.array([[1,1,0,2,2],[1,1,0,2,2]],np.uint16)
    probability = np.full(ids.shape,.5,np.float32)
    probability[ids==2] = np.nextafter(np.float32(.5),np.float32(1))
    expected,_ = semantic_crossover.revote_instances(ids,probability)
    assert post.vote_instances(ids,probability) == expected == {'1':0,'2':1}
    region = np.zeros((25,30),np.uint16)
    region[2:23,2:20] = 1
    region[5:8,21:24] = 2
    old,classes,_ = final_instance_cleanup.merge_final_instances(region,{'1':0,'2':1},200)
    new,new_classes = post.merge_final_instances(region,{'1':0,'2':1})
    assert np.array_equal(old,new) and classes == new_classes


def test_integer_ID_sort_and_cap_match_frozen_finish():
    source = np.array([[1,1,0,17,17,0,60000],[1,1,0,17,17,0,-1]],np.int32)
    expected = fast_instance_readout.finish_watershed(source,dict(seed_min_area=50,
        final_min_instance_area=200,final_small_instance_action='neighbor_merge_v1',max_instance_id=65535))[0]
    assert np.array_equal(post.finish_watershed(source),expected)


def test_image_letterbox_crop_and_affinity_fusion_exact():
    rgb = np.arange(73*101*3,dtype=np.uint8).reshape(73,101,3)
    expected,ph,pw = tensor_from_rgb(rgb,'cpu')
    actual,ah,aw = pipe.image_tensor(rgb,'cpu')
    assert (ph,pw)==(ah,aw) and torch.equal(actual,expected) and actual.stride()==expected.stride()
    output = torch.linspace(-2,2,128*128).reshape(1,1,128,128)
    assert torch.equal(pipe.crop_native(output,ph,pw,rgb.shape[:2]),
                       crop_letterbox_output(output,1024,ph,pw,rgb.shape[:2]))
    logits = torch.linspace(-8,8,8*20*24).reshape(1,8,20,24)
    expected = affinity_boundary_probability(logits,mode='gated',short_reduction='top2',
        distance2_weight=.5,distance4_weight=.25,support_threshold=.2,support_temperature=.05)
    assert torch.equal(pipe.affinity_boundary(logits),expected)


class FakeEngine:
    def __init__(self,role):
        self.role = role
        self.inputs = dict(trunk=['image'],semantic=[*pipe.FEATURES,'image'],affinity=list(pipe.FEATURES),
                           naf=['inputs'],d5a=['xt','condition','t_index'])[role]
        self.outputs = dict(trunk=list(pipe.FEATURES),semantic=['semantic_logits'],
                            affinity=['affinity_logits'],naf=['logits'],d5a=['x0'])[role]
        self.batches=[]
    def __call__(self,**inputs):
        assert set(inputs)==set(self.inputs)
        if self.role=='trunk': return {name:inputs['image'] for name in pipe.FEATURES}
        if self.role=='semantic': return {'semantic_logits':inputs['image'][:,:1]-.5}
        if self.role=='affinity': return {'affinity_logits':torch.full((1,8,128,128),5.)}
        if self.role=='d5a':
            assert inputs['xt'] is inputs['condition'] and inputs['t_index'].tolist()==[16]
            return {'x0':inputs['condition']+.01}
        self.batches.append(inputs['inputs'].shape[0])
        return {'logits':inputs['inputs']*2-1}


@pytest.mark.parametrize('shape',[(64,80),(1024,1224)])
def test_whole_T8_NAF_tail_batch_weighted_stitch_exact(shape):
    engines = {role:FakeEngine(role) for role in pipe.ROLES}
    pipeline = pipe.Pipeline(engines,device='cpu')
    boundary = np.linspace(0,1,shape[0]*shape[1],dtype=np.float32).reshape(shape)
    class Model(torch.nn.Module):
        input_mode='boundary'
        def forward(self,value): return (value*2-1).sigmoid()
    settings = deepcopy(infer_compact.naf.INPUT_CONTRACT)
    expected,_ = predict_boundary_rgb_tiled(boundary,np.zeros((*shape,3),np.uint8),Model(),settings,device='cpu')
    actual = pipeline._naf(boundary)
    assert np.array_equal(actual,expected)
    if shape==(1024,1224): assert engines['naf'].batches==[16,16,10]


def test_full_small_CPU_fake_engine_flow_output_stem_and_formats(tmp_path):
    engines = {role:FakeEngine(role) for role in pipe.ROLES}
    # 低NAF输出产生正常的大种子；不在此运行任何真实神经网络。
    class LowNAF(FakeEngine):
        def __call__(self,**inputs): return {'logits':torch.full_like(inputs['inputs'],-4.)}
    engines['naf']=LowNAF('naf')
    pipeline = pipe.Pipeline(engines,device='cpu')
    rgb=np.full((64,80,3),180,np.uint8)
    ids,classes=pipeline.predict(rgb)
    post.verify_output(ids,classes,rgb.shape[:2])
    assert pipeline.calls==dict(trunk=3,semantic=2,affinity=1,naf=1,d5a=2)
    pipe.save_prediction(tmp_path,'任意名字',ids,classes)
    png=cv2.imdecode(np.frombuffer((tmp_path/'任意名字_inst.png').read_bytes(),np.uint8),cv2.IMREAD_UNCHANGED)
    assert np.array_equal(png,ids) and png.dtype==np.uint16
    assert json.loads((tmp_path/'任意名字_class.json').read_text(encoding='utf8'))==classes


def test_generic_input_inventory_and_role_files_without_hash_locks(tmp_path):
    images=tmp_path/'images'; images.mkdir()
    (images/'non_test_name.png').write_bytes(b'not decoded by inventory')
    assert pipe.input_files(images)[0].stem=='non_test_name'
    (images/'non_test_name.jpg').write_bytes(b'duplicate stem')
    with pytest.raises(ValueError): pipe.input_files(images)
    rows={}
    for role in pipe.ROLES:
        (tmp_path/(role+'.engine')).write_bytes(b'weights')
        rows[role]={'file':role+'.engine'}
    (tmp_path/'engines.json').write_text(json.dumps({'engines':rows}))
    assert set(pipe.engine_files(tmp_path))==set(pipe.ROLES)
