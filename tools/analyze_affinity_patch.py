# -*- coding: utf-8 -*-
"""分类别汇总原生patch检查；不把缺标训练域或测试外观当官方得分。"""
from __future__ import annotations
import argparse,html,json,sys
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from utils.patch_diagnostic import tile_boxes
from tools.render_backend_ablation import text_line,write_png


def enrich_sizes(root):
    """只读原生实例文件。分档不改变任何实例，也不将小实例标为假阳性。"""
    cases=json.loads((root/'cases.json').read_text(encoding='utf8'));result=[]
    variants=['global_d5a','global_raw','patch1024_d5a','patch1024_raw','patch512_d5a','patch512_raw']
    bins=np.array([0,.05,.1,.25,.5,1,2,np.inf]);labels=['under_005','005_01','01_025','025_05','05_1','1_2','over_2']
    def prediction(case,variant):
        folder=root/case['name']/variant
        ids=cv2.imread(str(folder/(case['name']+'_inst.png')),cv2.IMREAD_UNCHANGED)
        classes=json.loads((folder/(case['name']+'_class.json')).read_text(encoding='utf8'))
        return ids,classes
    def class_areas(ids,classes,c):
        areas=np.bincount(ids.ravel());selected=[int(k) for k,v in classes.items() if v==c]
        return areas[selected]
    for case in cases:
        base,bc=prediction(case,'global_d5a');medians={c:float(np.median(class_areas(base,bc,c))) for c in [0,1]}
        for variant in variants:
            ids,classes=prediction(case,variant);entry=dict(source=case['name'],kind=case['kind'],variant=variant,phases={})
            for c,name in [(1,'ferrite'),(0,'pearlite')]:
                areas=class_areas(ids,classes,c);scaled=areas/medians[c]
                entry['phases'][name]=dict(control_median_area=medians[c],counts=np.histogram(scaled,bins)[0].tolist(),
                    pixels=np.histogram(scaled,bins,weights=areas)[0].astype(np.int64).tolist())
                assert sum(entry['phases'][name]['counts'])==len(areas)
                assert sum(entry['phases'][name]['pixels'])==int(areas.sum())
            result.append(entry)
        if case['kind']=='test':
            # 选择P实例数增量最多的固定512方窗，明确是失败风险定位，不是随机效果展示。
            candidate,cc=prediction(case,'patch1024_d5a');scores=[]
            for box in tile_boxes(base.shape,512,.5):
                y,x,y1,x1=box
                old=sum(bc[str(int(i))]==0 for i in np.unique(base[y:y1,x:x1]) if i)
                new=sum(cc[str(int(i))]==0 for i in np.unique(candidate[y:y1,x:x1]) if i)
                scores.append((new-old,box,old,new))
            score,box,old,new=max(scores,key=lambda x:x[0]);y,x,y1,x1=box
            image=cv2.imread(case['path'])[y:y1,x:x1];panels=[]
            for title,variant in [('Raw','raw'),('Global D5a','global_d5a'),('Native 1024','patch1024_d5a'),('Native 512 x2','patch512_d5a')]:
                if variant=='raw':body=image.copy()
                else:
                    ids,classes=prediction(case,variant);roi=ids[y:y1,x:x1]
                    rng=np.random.default_rng(1024);colors=rng.integers(40,235,size=(int(ids.max())+1,3),dtype=np.uint8);colors[0]=0
                    lut=np.zeros(int(ids.max())+1,np.bool_)
                    for key,c in classes.items():lut[int(key)]=c==0
                    body=np.full((*roi.shape,3),235,np.uint8);body[lut[roi]]=colors[roi[lut[roi]]]
                body=cv2.resize(body,(400,400),interpolation=cv2.INTER_NEAREST if variant!='raw' else cv2.INTER_AREA)
                body=cv2.copyMakeBorder(body,28,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255));text_line(body,title,(5,20),400,size=.5);panels.append(body)
            write_png(root/'gallery'/f'{case["name"]}_pearlite_ids.png',np.concatenate(panels,1))
            result.append(dict(source=case['name'],kind='pearlite_fragment_roi',box=box,global_pearlite_ids=old,patch1024_pearlite_ids=new,
                selection='maximum predicted P count increase among 512px windows; colors distinguish IDs within each panel only'))
    (root/'size_bands.json').write_text(json.dumps(dict(labels=labels,rows=result),ensure_ascii=False,indent=2),encoding='utf8')
    print('SIZE BANDS COMPLETE',len(result))


def analyze(root):
    report=json.loads((root/'report.json').read_text(encoding='utf8'));assert report['complete']
    rows=json.loads((root/'rows.json').read_text(encoding='utf8'));cases=json.loads((root/'cases.json').read_text(encoding='utf8'))
    variants=['global_d5a','global_raw','patch1024_d5a','patch1024_raw','patch512_d5a','patch512_raw']
    lookup={(r['source'],r['variant']):r for r in rows}
    assert len(lookup)==len(rows)==len(cases)*len(variants)
    summary=dict(scope=report['scope'],groups={},restoration={},batch=json.loads((root/'batch_check.json').read_text(encoding='utf8')))
    if (root/'size_bands.json').exists():summary['size_bands']=json.loads((root/'size_bands.json').read_text(encoding='utf8'))
    if (root/'structure/report.json').exists():summary['structure']=json.loads((root/'structure/report.json').read_text(encoding='utf8'))
    for kind in ['train','test']:
        group={}
        for variant in variants:
            selected=[r for r in rows if r['kind']==kind and r['variant']==variant]
            ratios=[];pix=[];counts=[]
            for r in selected:
                current=r['phases']['ferrite'];base=lookup[(r['source'],'global_d5a')]['phases']['ferrite']
                ratios.append(current['mean_area']/base['mean_area']);pix.append(current['pixels']/base['pixels']);counts.append(current['instances']/base['instances'])
            row=dict(images=len(selected),phases={name:{k:sum(r['phases'][name][k] for r in selected) for k in ['instances','pixels']} for name in ['ferrite','pearlite']},
                ferrite_mean_area_change_median=float(np.median(ratios)-1),ferrite_mean_area_change_range=[float(min(ratios)-1),float(max(ratios)-1)],
                ferrite_pixel_change_median=float(np.median(pix)-1),ferrite_count_change_median=float(np.median(counts)-1),
                ferrite_tiny_under200=sum(r['phases']['ferrite']['tiny_under200'] for r in selected))
            if kind=='train':
                row['known_domain']={}
                for phase in ['ferrite','pearlite']:
                    values=[r['known_domain'][phase] for r in selected]
                    totals={k:sum(v[k] for v in values) for k in ['gt_instances','pred_instances_in_known_domain','matches','iou_sum','split_gt','merged_pred']}
                    totals.update(matched_iou=totals['iou_sum']/max(1,totals['matches']),gt_penalized_iou=totals['iou_sum']/max(1,totals['gt_instances']))
                    row['known_domain'][phase]=totals
                row['new_splits_vs_global']={phase:sum(r.get('new_errors',{}).get('split_by_class',{}).get(phase,0) for r in selected) for phase in ['ferrite','pearlite']}
                row['new_merges_vs_global']={phase:sum(r.get('new_errors',{}).get('merge_by_class',{}).get(phase,0) for r in selected) for phase in ['FF','PP','mixed']}
            group[variant]=row
        summary['groups'][kind]=group
    reconstruction=json.loads((root/'restoration/rows.json').read_text(encoding='utf8'))
    for profile in dict.fromkeys(r['profile'] for r in reconstruction):
        selected=[r for r in reconstruction if r['profile']==profile];entry={'pairs':len(selected)}
        for key in ['rgb_mse','gradient_l1','edge_support_recall','novel_edge_fraction','edge_band_area_ratio']:
            entry[key]={name:float(np.mean([r[name][key] for r in selected])) for name in ['input','d5a']}
        if profile!='identity':
            for key in ['rgb_mse','gradient_l1']:
                entry[key]['paired_ratio_median']=float(np.median([r['d5a'][key]/max(1e-12,r['input'][key]) for r in selected]))
        summary['restoration'][profile]=entry
    (root/'analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    fig,axes=plt.subplots(1,2,figsize=(13,4.2),layout='constrained')
    x=np.arange(len(cases));names=[c['name'] for c in cases]
    for variant in variants[1:]:
        area=[];count=[]
        for c in cases:
            b=lookup[(c['name'],'global_d5a')]['phases']['ferrite'];a=lookup[(c['name'],variant)]['phases']['ferrite']
            area.append(100*(a['mean_area']/b['mean_area']-1));count.append(100*(a['instances']/b['instances']-1))
        axes[0].plot(x,area,'.-',label=variant);axes[1].plot(x,count,'.-',label=variant)
    for ax,title in zip(axes,['Ferrite mean area: change vs full D5a','Ferrite instance count: change vs full D5a']):
        ax.set_xticks(x,names,rotation=45,ha='right');ax.set_title(title);ax.set_ylabel('Percent');ax.axhline(0,color='gray',lw=.8);ax.grid(alpha=.15)
    axes[0].legend(fontsize=8);fig.savefig(root/'ferrite_changes.png',dpi=150);plt.close(fig)
    parts=['<!doctype html><meta charset="utf-8"><title>原生patch与D5a检查</title><style>body{max-width:1500px;margin:25px auto;font:16px/1.6 sans-serif;color:#25313a}img{width:100%}table{border-collapse:collapse}th,td{border:1px solid #ccd3d9;padding:7px}details{margin:16px 0;border:1px solid #ccd3d9;padding:10px}summary{cursor:pointer}</style>',
        '<h1>原生patch、D5a与分类别实例检查</h1><p>固定control e20、D5a e60及两路全图语义。所有分区只改变affinity的输入视野／修复；每图只做一次全局分水岭。未训练、未改变主线。</p>',
        '<p>训练匹配只限原标可信域，未知区域从双方移除，数值不是官方精度。训练源已被模型见过。测试图只有完整预测统计及目检，面积变化方向不代表正确性。global_raw也复用固定D5a全图几何语义，属于affinity单独改输入的控制。</p>']
    for kind,title in [('train','训练原标可信域'),('test','无标签测试预测')]:
        parts.append(f'<h2>{title}</h2><table><tr><th>方案</th><th>铁素体数</th><th>珠光体数</th><th>铁素体平均面积变化中位</th><th>铁素体小于200像素</th></tr>')
        for name,r in summary['groups'][kind].items():
            parts.append(f'<tr><td>{name}</td><td>{r["phases"]["ferrite"]["instances"]}</td><td>{r["phases"]["pearlite"]["instances"]}</td><td>{r["ferrite_mean_area_change_median"]:+.2%}</td><td>{r["ferrite_tiny_under200"]}</td></tr>')
        parts.append('</table>')
    parts.append('<img src="ferrite_changes.png"><h2>完整8图与固定局部</h2><p>金色为铁素体，蓝色为珠光体。两档分别展示原图、global_d5a、patch_D5a、patch_raw；所有固定样本均保留。</p>')
    for case in cases:
        name=case['name'];parts.append(f'<details><summary>{html.escape(name)} ({case["kind"]})</summary>')
        for size in [1024,512]:
            for suffix in ['full','upper','lower','boundary']:
                p=f'gallery/{name}_{size}_{suffix}.png';parts.append(f'<h3>{size} / {suffix}</h3><a href="{p}"><img loading="lazy" src="{p}"></a>')
        parts.append('</details>')
        identity=root/'gallery'/f'{name}_pearlite_ids.png'
        if identity.exists():
            parts.append(f'<details><summary>{name} 珠光体分片定位</summary><p>按1024窗口预测相对全图的P实例增量选出512像素局部。浅灰为非珠光体，彩色只区分本面板实例；不是跨方案配对颜色，也不是GT。用于暴露被同类着色遮住的小片。</p><img src="gallery/{identity.name}"></details>')
    parts.append('<h2>D5a已知清晰参照检查</h2><p>固定中心256像素原尺寸裁图；identity检查清晰输入被改变的程度，其余比较降采样/模糊后的输入与D5a输出。边缘支持率及假边比例是相对清晰图纹理的诊断，不是实例边界准确率。</p>')
    for case in [c for c in cases if c['kind']=='train']:
        parts.append(f'<details><summary>{case["name"]}</summary>')
        for p in sorted((root/'restoration').glob(case['name']+'*.png')):
            link=p.relative_to(root).as_posix();parts.append(f'<p>{p.stem}</p><img loading="lazy" src="{link}">')
        parts.append('</details>')
    if 'structure' in summary:
        parts.append('<h2>追加：已知铁素体内部的结构响应</h2><p>在原标可信、距实例边缘至少8个输出格的铁素体内部，统计清晰参照affinity边界概率低于0.45、退化输入或D5a恢复后高于0.65的格点。低分辨模糊档出现网状纹理后，对全部32对重放检查；不是测试GT或最终实例错误数。</p><table><tr><th>退化</th><th>插值输入新增强响应</th><th>D5a新增强响应</th></tr>')
        for profile,row in summary['structure']['summary'].items():
            parts.append(f'<tr><td>{profile}</td><td>{row["variants"]["input"]["new_high_from_clear_low"]}</td><td>{row["variants"]["d5a"]["new_high_from_clear_low"]}</td></tr>')
        parts.append('</table><p>以下上排为清晰/插值/D5a，下一排为对应固定0–1色阶的affinity边界概率。只列本地已下载例图；全部32对数字见structure/rows.json。</p>')
        for image in sorted((root/'structure').glob('*.png')):
            parts.append(f'<details><summary>{image.stem}</summary><img loading="lazy" src="structure/{image.name}"></details>')
    (root/'index.html').write_text('\n'.join(parts),encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--sizes',action='store_true');args=p.parse_args()
    if args.sizes:enrich_sizes(args.root.resolve())
    else:analyze(args.root.resolve())
