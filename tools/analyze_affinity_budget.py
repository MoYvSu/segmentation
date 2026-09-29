# -*- coding: utf-8 -*-
"""汇总固定8视图的梯度预算短测；所有图与逐图记录留在私有目录。"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]


def analyze(root):
    report=json.loads((root/'report.json').read_text(encoding='utf8'))
    initial=json.loads((root/'initial.json').read_text(encoding='utf8'))
    assert report['complete'] and len(initial)==8
    initial_bce={direction:float(np.mean([r['selected'][direction]['bce'] for r in initial
        if r['selected'][direction]['bce'] is not None])) for direction in ['same','different']}
    summary=dict(initial_selected_bce=initial_bce,results=report['results'],scope=report['scope'])
    fig,axes=plt.subplots(1,3,figsize=(14,3.6),layout='constrained')
    arms={'logit':'Old output budget','parameter':'15% parameter budget'}
    colors={'logit':'#457b9d','parameter':'#d1495b'}
    tables=[]
    for arm,label in arms.items():
        rows=read_rows(root/arm/'steps.jsonl');active=[r for r in rows if r['active']]
        assert len(rows)==64 and len(active)==16
        axes[0].plot([r['step'] for r in active],[r['details']['parameter_ratio']*100 for r in active],'.-',label=label,color=colors[arm])
        axes[1].plot([r['step'] for r in active],[r['details']['adamw_extra_delta_ratio']*100 for r in active],'.-',color=colors[arm])
        x=np.arange(1,9)
        axes[2].plot(x,[np.mean([r['manual_bce'] for r in rows[k:k+8]]) for k in range(0,64,8)],'.-',color=colors[arm])
        summary['results'][arm]['parameter_ratio_range']=[min(r['details']['parameter_ratio'] for r in active),max(r['details']['parameter_ratio'] for r in active)]
        summary['results'][arm]['gradient_cosine_median']=float(np.median([r['details']['gradient']['cosine'] for r in active]))
        summary['results'][arm]['adamw_prediction_max_error']=max(r['adamw_prediction_max_error'] for r in rows)
        result=report['results'][arm]
        tables.append(f'<tr><td>{label}</td><td>{result["parameter_ratio_median"]:.2%}</td>'
            f'<td>{result["adamw_extra_delta_ratio_median"]:.2%}</td>'
            f'<td>{result["fixes"]["merge"]["fixed"]}/{result["fixes"]["merge"]["cases"]}</td>'
            f'<td>{result["fixes"]["split"]["fixed"]}/{result["fixes"]["split"]["cases"]}</td>'
            f'<td>{result["new_split_pairs"]}</td><td>{result["new_merge_pairs"]}</td></tr>')
    for ax,title,ylabel,xlabel in zip(axes,
        ['Auxiliary / manual parameter gradient','Additional AdamW update at same history','Manual BCE: each complete 8-view cycle'],
        ['Percent','Percent','BCE'],['Optimizer step','Optimizer step','Cycle']):
        ax.set(title=title,ylabel=ylabel,xlabel=xlabel);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.savefig(root/'curves.png',dpi=160);plt.close(fig)
    (root/'analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    cards=[]
    for item in initial:
        key=item['source']+'_'+item['view']
        imgs=''.join(f'<h3>{label}</h3><a href="{arm}/monitor/step_064/{key}.png"><img loading="lazy" src="{arm}/monitor/step_064/{key}.png"></a>' for arm,label in arms.items())
        cards.append(f'<details><summary>{html.escape(key)}</summary>{imgs}</details>')
    page='''<!doctype html><meta charset="utf-8"><title>Affinity budget short probe</title>
<style>body{max-width:1450px;margin:25px auto;font:16px/1.6 sans-serif;color:#202b33}img{width:100%}table{border-collapse:collapse}th,td{border:1px solid #cbd5dc;padding:9px}details{margin:20px 0;border:1px solid #cbd5dc;padding:12px}summary{cursor:pointer}</style>
<h1>参数梯度预算短测</h1><p>同一control e20起点，4张训练源的原图／固定模糊共8视图，各64次更新。
固定特征、固定初始选位分区、同输入顺序；仅比较附加监督预算。此处是小样本过拟合机制检查，
不是留出验证或官方精度。旧预算分支也在进行额外训练，不能把它相对初始的变化归给新预算。</p>
<p>真实更新比例是在同一优化器历史上“本步加附加项／本步不加”的假想更新之差，
不是两条完整训练轨迹之差。修复与新增错误的点对采样不同，不相减为净准确率。</p>
<table><tr><th>分支</th><th>参数梯度比中位</th><th>附加AdamW更新比中位</th><th>修复粘连</th><th>修复误切</th><th>新增误切</th><th>新增粘连</th></tr>'''+''.join(tables)+'</table><img src="curves.png">'+''.join(cards)
    (root/'index.html').write_text(page,encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    analyze(parser.parse_args().root.resolve())
