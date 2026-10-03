# -*- coding: utf-8 -*-
"""原尺寸CPU marker反事实：只替换watershed种子，保持真实障碍图。

不读取模型或具体赛题图，不改原部署函数；输入来自已封存预测。
preseal、nothin与reflect只用预测边界生成种子，不使用GT补边或强制实例数量。
"""
from __future__ import annotations

import hashlib
from numbers import Integral
from unittest.mock import patch

import cv2
import numpy as np
import torch

from tools.affinity_connectivity_views import decode_native
from tools.probe_affinity_partition import _native_tensor, decode_capture
from tools.semantic_marker_diagnostic import anchored_marker_stages, marker_stages
from utils.post_process import _boundary_skeleton_belt


def _array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _markers(value, shape):
    value=np.asarray(value)
    if value.shape!=shape or value.dtype!=np.int32 or np.any(value<0):
        raise ValueError('Replacement marker must be nonnegative int32 with native HW shape')
    return value.copy()


def build_marker_variant(stages, mode, reflection_padding=32):
    """返回独立marker阶段候选；真实watershed高阈值barrier保持原样。"""
    if mode=='preseal':
        return anchored_marker_stages(stages)
    if mode not in ('nothin','reflect'):
        raise ValueError('Marker variant must be preseal, nothin or reflect')
    if mode=='reflect' and (isinstance(reflection_padding,bool)
            or not isinstance(reflection_padding,Integral) or reflection_padding<=0):
        raise ValueError('reflection_padding must be a positive integer')
    resolved=stages['resolved']
    source=np.asarray(stages['marker_boundary_mask'])
    if source.ndim!=2 or source.size==0:
        raise ValueError('Marker boundary mask must be nonempty HW')
    bridged=(source>0).astype(np.uint8)*255
    bridge=int(resolved['bridge_width']); dilate=int(resolved['watershed_dilate_width'])
    if bridge>0:
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*bridge+1,2*bridge+1))
        bridged=cv2.dilate(bridged,kernel)
    extras={}
    if mode=='reflect':
        padding=int(reflection_padding)
        # Use the project's BORDER_REFLECT convention, including the edge pixel.
        # Bridging occurs before extension, so the original native bridge is unchanged.
        padded=cv2.copyMakeBorder(bridged,padding,padding,padding,padding,cv2.BORDER_REFLECT)
        expanded_skeleton,_=_boundary_skeleton_belt(padded,0,0)
        h,w=source.shape
        skeleton=expanded_skeleton[padding:padding+h,padding:padding+w].copy()
        belt=skeleton.copy()
        extras=dict(reflection_padding=padding,reflection_mode='cv2.BORDER_REFLECT',
                    cropped_skeleton=skeleton,reflection_padding_stability='requires actual comparison; not assumed')
    else:
        belt=bridged.copy()
    if dilate>0:
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*dilate+1,2*dilate+1))
        belt=cv2.dilate(belt,kernel)
    before_seal=belt.copy()
    seal=max(0,int(resolved['marker_border_seal_width']))
    if seal:
        seal=min(seal,max(1,min(source.shape)//2))
        belt[:seal]=255; belt[-seal:]=255; belt[:,:seal]=255; belt[:,-seal:]=255
    cores=cv2.bitwise_not(belt)
    count,labels,stats,_=cv2.connectedComponentsWithStats(cores,connectivity=8)
    markers=np.zeros(source.shape,np.int32); valid=0
    for label in range(1,count):
        if int(stats[label,cv2.CC_STAT_AREA])<int(resolved['min_instance_area']):
            continue
        valid+=1; markers[labels==label]=valid
    return {**stages,'marker_boundary_mask':source.copy(),'marker_belt':belt,'cores':cores,
        'core_labels':labels,'marker_labels':markers,'marker_count':valid,
        'bridged_marker_mask':bridged,'dilated_marker_before_seal':before_seal,
        'resolved':{**resolved,'diagnostic_variant':'marker_'+mode+'_v1',
                    **({'reflection_padding':padding,'reflection_mode':'cv2.BORDER_REFLECT'} if mode=='reflect' else {})},
        'diagnostic_only':True,'real_barrier_changed':False,
        'change':('reflect bridged marker before actual thinning; crop then retain dilate/seal/min_area'
                  if mode=='reflect' else 'skip thinning of marker barrier only; retain bridge/dilate/seal/min_area'),
        **extras}


def decode_marker_replay(semantic, boundary, config, marker_labels=None):
    """通过真实decode_native后续流程，只在唯一watershed调用处替换marker。

    None复用decode_capture；原marker/elevation必须与原完整阶段复刻精确
    一致。返回original_stages与original_markers，避免把候选早期屏障
    冒充原部署。stages仅末端marker_labels跟随注入，其余为原阶段。
    本隔离入口要求原部署存在种子且关闭额外marker_partition_restore。
    """
    semantic=_native_tensor(semantic,'semantic'); boundary=_native_tensor(boundary,'boundary')
    if semantic.shape!=boundary.shape or bool(((boundary<0)|(boundary>1)).any()):
        raise ValueError('Native semantic/boundary shape or probability range differs')
    if config['inference'].get('marker_partition_restore',{}).get('enabled',False):
        raise ValueError('Marker replay requires disabled marker_partition_restore')
    base=marker_stages(boundary[0,0].numpy(),config)
    if not base['marker_count']:
        raise ValueError('Isolated marker replay requires original nonempty seeds and one watershed call')
    selected=base['marker_labels'].copy() if marker_labels is None else _markers(marker_labels,tuple(boundary.shape[-2:]))
    probability=semantic.sigmoid()[0,0].numpy()
    infer=config['inference']; sem_mask=probability>float(infer.get('threshold',.5))
    image=np.full((*sem_mask.shape,3),128,np.uint8); image[sem_mask]=200
    overlay=image.copy(); overlay[base['barrier_belt']>0]=255
    expected_image=cv2.addWeighted(image,.7,overlay,.3,0)
    real=cv2.watershed; observed=[]

    def capture(ws_image,ws_markers):
        if observed:
            raise AssertionError('Expected exactly one real watershed call')
        if not np.array_equal(ws_markers,base['marker_labels']):
            raise AssertionError('Actual original marker differs from full native stage replay')
        if not np.array_equal(ws_image,expected_image):
            raise AssertionError('Actual original watershed elevation/barrier differs')
        image_before=ws_image.copy(); original_marker=ws_markers.copy()
        actual_marker=original_marker.copy() if marker_labels is None else selected.copy()
        before=actual_marker.copy()
        result=real(ws_image,actual_marker)
        if not np.array_equal(ws_image,image_before):
            raise AssertionError('Watershed mutated the frozen elevation image')
        observed.append(dict(original_markers=original_marker,actual_markers=before,
            watershed=np.maximum(result,0).copy(),elevation_sha256=_array_sha(image_before),
            original_marker_sha256=_array_sha(original_marker),actual_marker_sha256=_array_sha(before),
            watershed_sha256=_array_sha(np.maximum(result,0))))
        return result

    with patch('utils.post_process.cv2.watershed',side_effect=capture):
        if marker_labels is None:
            value=decode_capture(semantic,boundary,config)
        else:
            instances,classes=decode_native(semantic,boundary,config)
            terminal={**base,'marker_labels':selected.copy(),
                'marker_count':int(np.count_nonzero(np.unique(selected)>0)),
                'diagnostic_only':True,'terminal_marker_override':True}
            value=dict(instances=instances,classes=classes,stages=terminal)
    if len(observed)!=1:
        raise AssertionError('Original decode must call real watershed exactly once')
    captured=observed[0]
    if (value['instances'].dtype!=np.uint16 or value['instances'].ndim!=2
            or value['instances'].shape!=tuple(boundary.shape[-2:]) or int(value['instances'].max())>65535
            or set(map(int,value['classes']))!=set(map(int,np.unique(value['instances'][value['instances']>0])))
            or any(c not in (0,1) for c in value['classes'].values())):
        raise AssertionError('Actual final uint16/instance-class contract differs')
    if marker_labels is None and (not np.array_equal(value['actual_markers'],captured['actual_markers'])
                                 or not np.array_equal(value['watershed'],captured['watershed'])):
        raise AssertionError('Original decode_capture and real watershed observation differ')
    value.update(original_stages=base,actual_markers=captured['actual_markers'],
        original_markers=captured['original_markers'],watershed=captured['watershed'],
        elevation_sha256=captured['elevation_sha256'],
        hashes={k:captured[k] for k in ('elevation_sha256','original_marker_sha256',
                                      'actual_marker_sha256','watershed_sha256')},
        capture_audit=dict(actual_watershed_calls=1,cpu_only=True,original_marker_exact=True,
            original_elevation_exact=True,real_barrier_changed=False,
            marker_override=marker_labels is not None,
            same_marker_as_baseline=np.array_equal(selected,base['marker_labels']),
            final_dtype='uint16',maximum_id=int(value['instances'].max()),
            stages_scope='original threshold/barrier stages; terminal seeds may be independently overridden'))
    return value
