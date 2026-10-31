# -*- coding: utf-8 -*-
"""恢复RGB缓存的CPU合同：同窗、同哈希、无量化与来源分组保护。"""
from types import SimpleNamespace
import json

import cv2
import numpy as np
import pytest
import torch

from tools.cache_boundary_rgb_restored import (
    array_sha, cache, file_sha, reference_hashes, restore_native_rgb,
    save_array, sealed_sources, tensor_state_sha, validate_rgb,
)
from utils.patch_diagnostic import BlendMap, tile_boxes


class FrozenRestorer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('gain', torch.tensor(.5))


def fake_ops(size=4):
    def tensor(rgb, device):
        assert rgb.shape==(size,size,3)
        return torch.from_numpy(rgb.copy()).permute(2,0,1).float()[None]/255.,0,0
    def restore(model,image,seed,mode,audit=None):
        assert mode=='zero'
        if audit is not None:
            audit.update(q_sample_calls=1,noise_nonzero=0,initial_equals_condition=True,step=16)
        result=image*model.gain
        return result,result
    def crop(value,actual_size,ph,pw,shape):
        assert actual_size==size and ph==pw==0 and tuple(shape)==(size,size)
        return value
    return SimpleNamespace(tensor_from_rgb=tensor,restore_probe=restore,
        crop_letterbox_output=crop,BlendMap=BlendMap,tile_boxes=tile_boxes)


def fixture():
    rgb=np.arange(6*7*3,dtype=np.uint8).reshape(6,7,3)
    model=FrozenRestorer();ops=fake_ops();windows=[]
    for index,box in enumerate(tile_boxes(rgb.shape[:2],4,.25)):
        y,x,y1,x1=box;part=rgb[y:y1,x:x1];value,_,_=ops.tensor_from_rgb(part,'cpu')
        result,_=ops.restore_probe(model,value,index,'zero')
        windows.append(dict(window=index,box=list(box),seed=index,
            input_tensor_sha256=array_sha(value.numpy()),restored_tensor_sha256=array_sha(result.numpy()),
            rgb1024_sha256=array_sha(part)))
    return rgb,model,ops,windows


def test_same_native_windows_blend_lossless_rgb():
    rgb,model,ops,windows=fixture();before=tensor_state_sha(model)
    output,proof,blend=restore_native_rgb(rgb,windows,model,ops,device='cpu',native_size=4,repeat_first=True)
    # BlendMap是同N1浮点累计，不承诺与一个全图独立算术逐位相同。
    np.testing.assert_allclose(output,rgb.astype(np.float32)/255*.5,atol=6e-8,rtol=0)
    assert output.shape==rgb.shape and output.dtype==np.float32
    assert proof[0]['same_window_repeat_exact'] is True
    assert all(p['actual_local_zero']['noise_nonzero']==0 for p in proof)
    assert all(v['covered_fraction']==1 for v in blend)
    assert tensor_state_sha(model)==before


@pytest.mark.parametrize('key', ['input_tensor_sha256','restored_tensor_sha256','rgb1024_sha256'])
def test_old_n1_hash_mismatch_rejected(key):
    rgb,model,ops,windows=fixture();windows[0][key]='0'*64
    with pytest.raises((RuntimeError,ValueError)):
        restore_native_rgb(rgb,windows,model,ops,device='cpu',native_size=4)


def test_wrong_window_order_and_count_rejected():
    rgb,model,ops,windows=fixture()
    with pytest.raises(ValueError):restore_native_rgb(rgb,windows[:-1],model,ops,device='cpu',native_size=4)
    windows[0],windows[1]=windows[1],windows[0]
    with pytest.raises(ValueError):restore_native_rgb(rgb,windows,model,ops,device='cpu',native_size=4)


def test_monitor_and_train_reference_key_names():
    a='a'*64;b='b'*64
    assert reference_hashes(dict(input_tensor_sha256=a,restored_tensor_sha256=b))==(a,b)
    assert reference_hashes(dict(input_sha256=a,restored_sha256=b))==(a,b)
    with pytest.raises(ValueError):reference_hashes(dict(input_sha256='bad',restored_sha256=b))


def test_zero_noise_actual_audit_required():
    rgb,model,ops,windows=fixture();old=ops.restore_probe
    def invalid(*a,**kw):
        value=old(*a,**kw)
        if kw.get('audit') is not None:kw['audit']['noise_nonzero']=1
        return value
    ops.restore_probe=invalid
    with pytest.raises(RuntimeError,match='零噪声'):
        restore_native_rgb(rgb,windows,model,ops,device='cpu',native_size=4)


def test_float32_saved_without_uint8_quantization(tmp_path):
    a=np.array([.12345679,.8765432,.000014],np.float32).reshape(1,1,3)
    p=tmp_path/'rgb.npy';row=save_array(p,a)
    assert np.array_equal(np.load(p,allow_pickle=False),a)
    assert row['restored_array_sha256']==array_sha(a) and row['restored_sha256']==file_sha(p)
    assert row['dtype']=='float32' and row['shape']==[1,1,3]
    assert np.any(np.rint(a*255)/255!=a)
    with pytest.raises(ValueError):save_array(p,a)
    with pytest.raises(ValueError):save_array(tmp_path/'bad.npy',a.astype(np.float16))


def test_original_rgb_shape_dtype_contract():
    a=np.zeros((4,5,3),np.uint8)
    assert np.array_equal(validate_rgb(a,[4,5]),a)
    for bad in [a.astype(np.float32),a[...,0],a[...,:2]]:
        with pytest.raises(ValueError):validate_rgb(bad)
    with pytest.raises(ValueError):validate_rgb(a,[5,4])


def test_sealed_source_manifest_checks_bytes_and_boundaries(tmp_path):
    p=tmp_path/'helper.py';p.write_text('x=1\n')
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(files=[dict(path='helper.py',sha256=file_sha(p))])))
    digest=file_sha(manifest);assert sealed_sources(tmp_path,digest)['checked_files']==1
    p.write_text('x=2\n')
    with pytest.raises(RuntimeError):sealed_sources(tmp_path,digest)
    manifest.write_text(json.dumps(dict(files=[dict(path='../outside.py',sha256='0'*64)])))
    with pytest.raises(RuntimeError):sealed_sources(tmp_path,file_sha(manifest))


def test_existing_output_never_overwritten(tmp_path):
    args=SimpleNamespace(output_root=tmp_path,code_root=tmp_path)
    with pytest.raises(FileExistsError):cache(args)

