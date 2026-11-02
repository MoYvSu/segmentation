# -*- coding: utf-8 -*-
"""复用一次marker阶段的S0读出；只省重复构建和随后被raw票覆盖的临时分类。"""
from __future__ import annotations

import hashlib

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from tools.semantic_marker_diagnostic import marker_stages
from tools.probe_affinity_marker_replay import build_marker_variant
from utils.instance_area_policy import resolve_instance_area_policy
from utils.marker_seed_support import filter_unsupported_seeds
from utils.post_process import _merge_region_map_to_cap
from utils.affinity_deployment import probability_to_logit

FORMAT = 'compact_S0_instance_readout_v1'


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _validate(semantic, after, reference, config, geometry_mode):
    if geometry_mode not in ('C','F0'):
        raise ValueError('geometry_mode must be C or F0')
    if (not torch.is_tensor(semantic) or semantic.ndim != 4 or tuple(semantic.shape[:2]) != (1,1)
            or semantic.dtype != torch.float32 or not torch.isfinite(semantic).all()):
        raise ValueError('Expected finite native FP32 B1HW semantic logits')
    shape=tuple(semantic.shape[-2:])
    for name,value in (('after',after),('reference',reference)):
        if (not isinstance(value,np.ndarray) or value.shape != shape or value.dtype != np.float32
                or not np.isfinite(value).all() or np.any((value<0)|(value>1))):
            raise ValueError(f'{name} must be native finite FP32 HW probability')
    infer=config['inference']
    if (infer.get('marker_partition_restore',{}).get('enabled',False)
            or infer.get('center_seeds',False) or infer.get('use_center_seeds',False)
            or float(infer.get('sem_edge_merge_weight',0)) != 0
            or float(infer.get('sem_edge_boost_alpha',0)) != 0):
        raise ValueError('Compact readout requires no semantic edge/center/partition-restoration branch')
    if (infer.get('semantic_vote_mode') != 'probability_mean'
            or infer.get('semantic_vote_erode_width',0) != 0
            or infer.get('semantic_vote_threshold',.5) != .5):
        raise ValueError('Temporary classes may be omitted only before the unchanged raw probability-mean revote')
    if geometry_mode=='F0' and torch.count_nonzero(semantic) != 0:
        raise ValueError('F0 requires exact zero geometry logits')
    maximum=int(infer.get('max_instance_id',65535))
    if not 1<=maximum<=65535:
        raise ValueError('max_instance_id must be within [1, 65535]')
    return semantic.detach().cpu().contiguous(), maximum


def finish_watershed(watershed, inference):
    """原Step6/7整数规则与稳定排序；不计算必被外层raw票覆盖的类别均值。"""
    if not isinstance(watershed,np.ndarray) or watershed.dtype != np.int32 or watershed.ndim != 2 or not watershed.size:
        raise ValueError('Expected nonempty HW watershed labels')
    source=np.maximum(watershed,0).copy()
    area=resolve_instance_area_policy(inference)
    maximum=int(inference.get('max_instance_id',65535))
    if not 1<=maximum<=65535:
        raise ValueError('max_instance_id must be within [1, 65535]')
    # 运算顺序刻意沿用原实现；本轮不叠加bincount/向量化重编号优化。
    candidate=np.zeros(source.shape,np.int32)
    for label in np.unique(source[source>0]):
        mask=source==label
        if int(mask.sum()) < area['pre_vote_final_min_area']:
            continue
        candidate[mask]=int(label)
    candidate,cap_merges=_merge_region_map_to_cap(candidate,maximum)
    labels,counts=np.unique(candidate[candidate>0],return_counts=True)
    ordered=[int(label) for label,_ in sorted(zip(labels,counts),key=lambda item:int(item[1]),reverse=True)]
    instances=np.zeros(source.shape,np.uint16)
    for current,label in enumerate(ordered,start=1):
        instances[candidate==label]=current
    # 仅为旧返回接口保留合法占位类别；入口必须随后调用原raw revote再final200。
    placeholders={current:0 for current in range(1,len(ordered)+1)}
    if int(instances.max())>maximum or len(placeholders)>maximum:
        raise RuntimeError('Compact watershed output exceeded the configured instance cap')
    return instances,placeholders,dict(pre_vote_final_min_area=area['pre_vote_final_min_area'],
        maximum_id=maximum,cap_merge_edges=[list(pair) for pair in cap_merges],
        sorting='original stable descending area; equal areas retain ascending original label',
        temporary_semantic_vote_calls=0,temporary_classes='P0 placeholders; mandatory raw revote before final cleanup')


@torch.no_grad()
def compact_supported_readout(semantic, after, reference, config, *, geometry_mode='C'):
    semantic,maximum=_validate(semantic,after,reference,config,geometry_mode)
    stages=marker_stages(after,config)
    reflected=build_marker_variant(stages,'reflect',32)
    original=reflected['marker_labels']
    selected,audit=filter_unsupported_seeds(original,reference,low_threshold=.45)
    retained=np.isin(original,audit['retained_ids'])&(original>0)
    if not np.array_equal(selected[retained],original[retained]) or np.any(selected[~retained] != 0):
        raise RuntimeError('S0 changed retained marker IDs/positions or introduced seed pixels')
    # 原_supported_readout先做S0，再由decode_marker_replay拒绝原始无种子输入。
    # 保留此失败顺序；不静默切到empty/fallback或保留最后一个seed。
    if not stages['marker_count']:
        raise ValueError('Isolated marker replay requires original nonempty seeds and one watershed call')
    # 保留decode_native真实的cat→同尺寸interpolate→2D sigmoid布局和顺序。
    # 不由单独sigmoid近似相等推断阈值mask完全相同。
    boundary=probability_to_logit(torch.from_numpy(after)[None,None])
    packed=F.interpolate(torch.cat([semantic,boundary],dim=1),size=after.shape,
                         mode='bilinear',align_corners=True)
    probability=torch.sigmoid(packed[0,0]).numpy()
    semantic_mask=probability>float(config['inference'].get('threshold',.5))
    image=np.full((*after.shape,3),128,np.uint8)
    image[semantic_mask]=200
    overlay=image.copy();overlay[stages['barrier_belt']>0]=255
    image=cv2.addWeighted(image,.7,overlay,.3,0)
    elevation_sha=array_sha(image)
    selected_sha=array_sha(selected)
    # cv2.watershed原地改marker，因此传副本；边界/语义/原marker均不改写。
    watershed=cv2.watershed(image,selected.copy())
    if array_sha(image) != elevation_sha or array_sha(selected) != selected_sha:
        raise RuntimeError('Watershed modified immutable elevation/selected markers')
    watershed=np.maximum(watershed,0)
    instances,classes,finish=finish_watershed(watershed,config['inference'])
    hashes=dict(elevation_sha256=elevation_sha,
        original_marker_sha256=array_sha(stages['marker_labels']),
        actual_marker_sha256=selected_sha,watershed_sha256=array_sha(watershed))
    capture=dict(actual_watershed_calls=1,cpu_only=True,original_marker_exact=True,
        original_elevation_exact=True,real_barrier_changed=False,marker_override=True,
        same_marker_as_baseline=bool(np.array_equal(selected,stages['marker_labels'])),
        final_dtype='uint16',maximum_id=int(instances.max()),
        stages_scope='one original stage construction; direct S0 markers/elevation, no diagnostic replay',
        original_marker_exact_basis='same shared marker_stages output reused; compatibility checked separately',
        original_elevation_exact_basis='constructed original formula; real WS input hash unchanged')
    audit.update(mode='original-zero',reference='original_N1_pre_NAF_boundary',
        reference_raw_array_sha256=array_sha(reference),marker_source='NAF_probability_actual_reflect32_markers',
        marker_reflection_padding=32,retained_marker_pixels_and_IDs_exact=True,
        readout_audit=capture,readout_hashes=hashes,
        compact_readout=dict(format=FORMAT,geometry_mode=geometry_mode,marker_stage_calls=1,
            reflect_marker_calls=1,S0_filter_calls=1,actual_WS_calls=1,
            raw_vote_required=True,final200_performed=False,
            **finish))
    if geometry_mode=='F0':
        audit['flat_semantic_proof']=dict(candidate='F0',semantic_base_uint8=128,
            geometry_semantic_logit_constant=0.,geometry_semantic_logits_zero=True,
            expected_WS_elevation_sha256=elevation_sha,actual_WS_elevation_sha256=elevation_sha,
            actual_WS_elevation_exact=True,actual_WS_calls=1,
            original_barrier_belt_sha256=array_sha(stages['barrier_belt']),
            barrier_overlay='cv2.addWeighted(base128,.7,base128_with_original_barrier255,.3,0)')
    return (instances,classes),audit
