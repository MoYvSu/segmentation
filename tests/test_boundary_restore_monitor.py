# -*- coding: utf-8 -*-
import random

import numpy as np
import torch

from tools.boundary_restore_monitor import paired_metrics, preserved_state, panels, zoom_box


def test_zoom_contains_a_real_gap_when_gaps_are_disconnected():
    target = np.zeros((512,512), np.float32)
    valid = np.ones_like(target, bool)
    gap = np.zeros_like(valid); gap[40:45,40:45] = True; gap[460:465,460:465] = True
    noise = np.zeros_like(valid)
    (x,y,x1,y1), label = zoom_box(target,valid,gap,noise)
    assert label == 'Gap' and gap[y:y1,x:x1].any()


def test_masked_metrics_use_actual_denominators():
    x = np.zeros((4,4), np.float32)
    y = np.ones_like(x)
    valid = np.zeros_like(x, bool); valid[:2] = True
    gap = valid.copy(); gap[0] = False
    noise = ~valid
    z = x.copy(); z[gap] = 1
    result = paired_metrics(x,y,z,valid,gap,noise)
    assert result['valid_pixels'] == 8 and result['gap_pixels'] == 4
    assert result['noise_pixels'] == 0 and result['noise_after_mae'] is None
    assert result['valid_after_mae'] == .5 and result['gap_after_mae'] == 0


def test_monitor_does_not_consume_rng_or_mode():
    model = torch.nn.Conv2d(1,1,1).train()
    random.seed(7); np.random.seed(8); torch.manual_seed(9)
    expected = (random.random(), np.random.rand(), torch.rand(1))
    random.seed(7); np.random.seed(8); torch.manual_seed(9)
    with preserved_state(model):
        assert not model.training
        random.random(); np.random.rand(); torch.rand(3)
    actual = (random.random(), np.random.rand(), torch.rand(1))
    assert model.training and expected[:2] == actual[:2]
    assert torch.equal(expected[2],actual[2])


def test_fixed_scale_panels_render(tmp_path):
    from PIL import Image
    output = tmp_path/'monitor.jpg'
    panels(output,'synthetic fixture',[('one',np.zeros((16,32,3),np.uint8)),
           ('two',np.full((32,16,3),255,np.uint8))],width=64)
    assert Image.open(output).size == (128,228)


def test_full_monitor_synthetic32_final_readout_rng_and_interrupted_resume(tmp_path):
    """实际走完整回调与旧解码；32独立命名合成源，不读取比赛数据或权重。"""
    import hashlib
    import json
    import cv2
    import yaml
    from models.boundary_restore import BoundaryRestoreNet, state_sha256
    from tools.boundary_restore_monitor import monitor
    from tools.boundary_restore_readout import evaluate_frozen_readout

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    source_dir=tmp_path/'synthetic_sources'
    source_dir.mkdir()
    records=[]
    for index in range(32):
        name=f'train_{index+1:03d}'
        rgb=np.full((80,96,3),90+index,np.uint8)
        rgb[:,48:]=180+index
        ids=np.ones((80,96),np.int32)
        ids[:,42+index%8:]=2
        ids[10:18,10:18]=0
        original=ids>0
        original[35:39,40:48]=False
        image=source_dir/(name+'.png')
        gt=source_dir/(name+'_gt.npz')
        classes=source_dir/(name+'_class.json')
        assert cv2.imwrite(str(image),rgb)
        np.savez_compressed(gt,instance_map=ids,original_covered=original,
                            filled=(ids>0)&~original,residual_unknown=ids==0)
        classes.write_text(json.dumps({'1':0,'2':1}),encoding='utf-8')
        files=dict(image=image,gt=gt,classes=classes)
        records.append(dict(source=name,**{k:str(p) for k,p in files.items()},
                            hashes={k+'_sha256':sha(p) for k,p in files.items()}))
    manifest=tmp_path/'processed32.json'
    manifest.write_text(json.dumps(dict(rows=records)),encoding='utf-8')

    boundary=np.full((96,112),.03,np.float32)
    boundary[:,48:57]=.9
    semantic=np.linspace(-2,2,112,dtype=np.float32)[None,None,None].repeat(96,axis=2)
    raw=np.full(boundary.shape,.8,np.float32)
    raw[:,56:]=.2
    frozen={'inference':dict(boundary_threshold=.65,marker_boundary_low_threshold=.45,
        marker_boundary_reconstruction_steps=8,marker_border_seal_width=2,bridge_width=1,
        watershed_dilate_width=1,min_instance_area=50,max_instance_id=65535,
        threshold=.5,semantic_vote_mode='probability_mean')}
    ids,classes,_=evaluate_frozen_readout(semantic,raw,boundary,frozen)
    frozen_path=tmp_path/'frozen.yaml'
    frozen_path.write_text(yaml.safe_dump(frozen),encoding='utf-8')
    cache=tmp_path/'synthetic_monitor.npz'
    np.savez_compressed(cache,rgb=np.full((*boundary.shape,3),137,np.uint8),
                        boundary=boundary,semantic=semantic,raw_probability=raw,instances=ids)
    labels=tmp_path/'synthetic_monitor_class.json'
    labels.write_text(json.dumps(classes),encoding='utf-8')
    control=tmp_path/'controls.json'
    control.write_text(json.dumps(dict(complete=True,config=str(frozen_path),config_sha256=sha(frozen_path),
        files=[dict(source='synthetic_monitor',path=str(cache),sha256=sha(cache),
                    control_classes=str(labels),control_classes_sha256=sha(labels))])),encoding='utf-8')
    config=dict(paths=dict(project_root=str(tmp_path)),boundary_restore=dict(
        manifest=str(manifest),crop_size=64,views_per_source=8,seed=20261025,
        identity_probability=.2,gap_probability=.6,noise_probability=.2,target_sigma=2.,valid_margin=8,
        monitor=dict(pair_indices=[0,1],thumbnail_width=64,tile_overlap=16,control_manifest=str(control))))
    old_threads=torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model=BoundaryRestoreNet(width=4,seed=11).train()
        before=state_sha256(model)
        frozen_sha={str(p):sha(p) for p in [manifest,cache,labels,control,frozen_path]}
        random.seed(14);np.random.seed(15);torch.manual_seed(16)
        expected=(random.random(),np.random.rand(),torch.rand(2))
        random.seed(14);np.random.seed(15);torch.manual_seed(16)
        result=monitor(model,config,tmp_path/'output',0,'cpu')
        actual=(random.random(),np.random.rand(),torch.rand(2))
        assert expected[:2]==actual[:2] and torch.equal(expected[2],actual[2])
        assert model.training and state_sha256(model)==before
        row=result['rows'][-1]
        assert row['kind']=='unlabelled_test_monitor' and row['no_test_gt']
        assert row['identity_epoch_exact'] and row['output_change_mae']==0
        assert len(result['rows'])==3
        assert all(r['valid_before_mae']==r['valid_after_mae'] for r in result['rows'][:2])
        assert (tmp_path/'output/monitor/e000/pair_000.jpg').is_file()
        assert (tmp_path/'output/monitor/e000/synthetic_monitor_whole.jpg').is_file()
        # 覆盖刚写过程图但checkpoint尚未记录完成的中断窗口：保留旧目录再重绘。
        repeated=monitor(model,config,tmp_path/'output',0,'cpu')
        assert repeated['interrupted_backup'] is not None
        assert repeated['rows']==result['rows']
        assert state_sha256(model)==before and model.training
        assert all(sha(__import__('pathlib').Path(p))==value for p,value in frozen_sha.items())
    finally:
        torch.set_num_threads(old_threads)
