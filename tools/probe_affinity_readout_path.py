# -*- coding: utf-8 -*-
"""真实融合边界驱动的合法短关系瓶颈反事实；不训练、不改变部署。

复用原七源探针的模型身份、透明捕获、原生合法支持与完整读出。
唯一改变为GT纠正位置：节点容量来自实际CUDA融合边界，目标通道须
在节点处参与top2（正）或为合法跨GT伴随关系（负），不是扫整界面。
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from tools import probe_affinity_bottleneck as original
from tools.run_marker_reflect import array_sha,read,write
from train_backend_adaptation import fusion_options
from utils.affinity_fusion import affinity_boundary_probability
from utils.affinity_readout_path import readout_bottleneck_targets


def make_selector(original_select):
    """原生域、端点及合法边检查完全复用；只替换网格选位函数。"""
    def select(tile,native_gt,decoded,case,probabilities):
        # 三个干预臂重用tiles但会重置mask/p；不能沿用上一个臂的grid。
        current=array_sha(probabilities)
        matched=current==tile.get('readout_current_short_sha256')
        grid=(tile['readout_current_grid'] if matched else tile['grid'])[0,0]

        def choose(p,gt,valid,start,end,direction,**kwargs):
            targets=readout_bottleneck_targets(grid,p,gt,valid,start,end,direction,**kwargs)
            # 原selector与轮次循环还读取edges/capacity/zero_capacity等路径字段。
            value=dict(targets['capacity_path']); value.update(targets)
            # 原runner的写mask接口；明确区分图容量瓶颈和实际修正关系。
            value['graph_bottleneck_indices']=targets['capacity_path'].get('bottleneck_indices',[])
            value['bottleneck_indices']=value.get('target_indices',[])
            value.update(selection_variant='fused_readout_v1',fused_grid_sha256=array_sha(grid),
                current_short_probabilities_sha256=array_sha(p),
                graph_probability_source='min(1-actual_fused_grid[a],1-actual_fused_grid[b]); not raw canonical affinity',
                correction_source='GT-legal actual active short channels or cross-GT companion channels')
            return value

        with patch.object(original,'constrained_bottleneck_path',side_effect=choose):
            return original_select(tile,native_gt,decoded,case,probabilities)
    return select


def make_recompute(original_recompute):
    """正式拼接完成后封存本轮真实grid，供下一轮择路；返回值不变。"""
    def recompute(views,tiles,masks,config,shape,**kwargs):
        result=original_recompute(views,tiles,masks,config,shape,**kwargs)
        mode,options=fusion_options(config)
        with torch.no_grad():
            for tile,(negative,positive),probabilities in zip(tiles,masks,result[2]):
                logits=original.corrected_logits(tile['logits'],negative,positive)
                grid=affinity_boundary_probability(logits.float(),mode=mode,**options)
                value=grid.cpu().numpy().copy()
                if not negative.any() and not positive.any() and not np.array_equal(value,tile['grid']):
                    raise RuntimeError('Readout selector empty-grid capture differs from actual original CUDA')
                tile['readout_current_grid']=value
                tile['readout_current_short_sha256']=array_sha(probabilities)
        return result
    return recompute


def run(args):
    selected=original.select_bottlenecks; recompute=original.recompute
    with patch.object(original,'select_bottlenecks',side_effect=make_selector(selected)), \
            patch.object(original,'recompute',side_effect=make_recompute(recompute)):
        original.run(args)
    output=ROOT/args.output
    report=read(output/'report.json')
    if not report.get('complete') or not report.get('frozen_states_exact'):
        raise RuntimeError('Actual complete frozen deployment checks required')
    report.update(format='affinity_readout_path_probe_v1',selection_variant='fused_readout_v1',
        only_variable='GT correction targets selected on actual fused CUDA grid; original model/fusion/thresholds/readout unchanged',
        independent_accuracy=False,training=False,
        interpretation='Constrained grid maximin is a surrogate, not native minimum-cut or watershed correctness')
    write(output/'report.json',report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True); p.add_argument('--output',default='outputs/readout_path')
    p.add_argument('--receipt',default='outputs/affinity_weak/capture.json')
    p.add_argument('--rounds',type=int,default=8); p.add_argument('--limit',type=int,default=0)
    run(p.parse_args())


if __name__=='__main__': main()
