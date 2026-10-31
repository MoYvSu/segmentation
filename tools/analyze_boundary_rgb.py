# -*- coding: utf-8 -*-
"""固定 RGB 模糊配对的 N0 响应诊断；不读取 GT、不训练、不拟合阈值。

统计原强响应丢失、原低响应新增，以及原 N0 闭合背景区域的粘连。
这些是原 N0 参考下的连续响应/拓扑变化，不是真实晶界正确率或面积分数。
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


PROTOCOL = {
    'strong_threshold': .65,
    'background_threshold': .45,
    'low_reference_threshold': .20,
    'background_connectivity': 8,
    'anchor_min_inside_distance_native': 4.,
    'anchor_grid_cells_per_axis': 4,
    'minimum_anchors_per_region': 3,
    'adjacency_halfwidth_native': 16.,
    'merge_rule': 'All sampled interior anchors of each original region must survive and share one candidate background component; two such adjacent regions with the same component count as merged',
}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def _probability(value):
    value=np.asarray(value)
    if value.ndim!=2 or value.dtype.kind!='f' or not np.isfinite(value).all() or np.any((value<0)|(value>1)):
        raise ValueError('Native probability must be finite floating HW in [0,1]')
    return value.astype(np.float32,copy=False)


def _divide(numerator,denominator):
    return float(numerator/denominator) if denominator else None


def _inside_distance(mask):
    return distance_transform_edt(np.pad(mask,1,mode='constant'))[1:-1,1:-1]


def _select_anchors(region, eligible, inside):
    """每个空间格取最深的一点；候选输出不参与选点，避免只挑单个中心。"""
    ys,xs=np.nonzero(region)
    y0,y1,x0,x1=int(ys.min()),int(ys.max())+1,int(xs.min()),int(xs.max())+1
    cells=PROTOCOL['anchor_grid_cells_per_axis']
    ey,ex=np.nonzero(eligible)
    gy=np.minimum(cells-1,(ey-y0)*cells//(y1-y0))
    gx=np.minimum(cells-1,(ex-x0)*cells//(x1-x0))
    points=[]
    for cell in range(cells*cells):
        indexes=np.flatnonzero(gy*cells+gx==cell)
        if indexes.size:
            index=indexes[np.argmax(inside[ey[indexes],ex[indexes]])]
            points.append([int(ey[index]),int(ex[index])])
    return points


def build_reference_regions(clean):
    """原 N0 闭合背景分量及邻接对；无 GT，也不把它当最终实例读出。"""
    clean=_probability(clean)
    background=clean<PROTOCOL['background_threshold']
    count,labels,stats,_=cv2.connectedComponentsWithStats(background.astype(np.uint8),connectivity=PROTOCOL['background_connectivity'])
    edge_ids=set(np.unique(np.r_[labels[0],labels[-1],labels[:,0],labels[:,-1]]).tolist())
    inside=_inside_distance(background)
    closed,regions,rejected=[],{},[]
    for region_id in range(1,count):
        if region_id in edge_ids:
            continue
        closed.append(region_id)
        region=labels==region_id
        eligible=region&(clean<PROTOCOL['low_reference_threshold'])&(inside>=PROTOCOL['anchor_min_inside_distance_native'])
        anchors=_select_anchors(region,eligible,inside) if eligible.any() else []
        row=dict(region_id=region_id,pixels=int(stats[region_id,cv2.CC_STAT_AREA]),
                 reliable_interior_pixels=int(eligible.sum()),anchors_yx=anchors)
        if len(anchors)<PROTOCOL['minimum_anchors_per_region']:
            rejected.append({**row,'reason':'too_few_spatially_distinct_deep_low_response_anchors'})
        else:
            regions[region_id]=row
    # 将背景标签最近邻扩展到原响应带；仅在固定窄带内取Voronoi相邻标签，
    # 避免把远隔大块响应的两区无条件当邻居。它只用于选择比较对。
    if background.any():
        distance,nearest=distance_transform_edt(~background,return_indices=True)
        expanded=labels[tuple(nearest)]
        near=distance<=PROTOCOL['adjacency_halfwidth_native']
        pairs=set()
        for first,second,allowed in [
            (expanded[:-1],expanded[1:],near[:-1]&near[1:]),
            (expanded[:,:-1],expanded[:,1:],near[:,:-1]&near[:,1:]),
        ]:
            change=allowed&(first!=second)&(first>0)&(second>0)
            for a,b in np.unique(np.stack([first[change],second[change]],axis=1),axis=0):
                if int(a) in regions and int(b) in regions:
                    pairs.add(tuple(sorted((int(a),int(b)))))
    else:
        pairs=set()
    return dict(labels=labels,regions=regions,pairs=sorted(pairs),
                summary=dict(background_components=count-1,closed_regions=len(closed),
                             anchor_eligible_regions=len(regions),anchor_ineligible_regions=len(rejected),
                             adjacent_pair_count=len(pairs),excluded_border_region_count=len(edge_ids-{0})),
                rejected_regions=rejected)


def _topology(reference,candidate):
    _,labels=cv2.connectedComponents((candidate<PROTOCOL['background_threshold']).astype(np.uint8),connectivity=PROTOCOL['background_connectivity'])
    rows,component_by_region={},{}
    for region_id,region in reference['regions'].items():
        anchors=np.asarray(region['anchors_yx'],dtype=np.int32)
        observed=labels[anchors[:,0],anchors[:,1]]
        unique=np.unique(observed)
        stable=len(unique)==1 and unique[0]>0
        component=int(unique[0]) if stable else None
        component_by_region[region_id]=component
        rows[region_id]=dict(anchor_count=len(anchors),anchors_still_background=int((observed>0).sum()),
                            unique_background_components=np.unique(observed[observed>0]).tolist(),
                            all_anchors_in_one_background=bool(stable),candidate_component=component)
    pairs=[]
    for first,second in reference['pairs']:
        a,b=component_by_region[first],component_by_region[second]
        state='ambiguous' if a is None or b is None else ('merged' if a==b else 'separate')
        pairs.append(dict(reference_regions=[first,second],candidate_components=[a,b],state=state))
    merged=sum(p['state']=='merged' for p in pairs)
    resolved=sum(p['state']!='ambiguous' for p in pairs)
    return dict(status='measured' if pairs else 'no_eligible_reference_pairs',
                total_pairs=len(pairs),resolved_pairs=resolved,merged_pairs=merged,
                separate_pairs=resolved-merged,ambiguous_pairs=len(pairs)-resolved,
                merged_fraction_all_pairs=_divide(merged,len(pairs)),
                merged_fraction_resolved_pairs=_divide(merged,resolved),
                stable_regions=sum(r['all_anchors_in_one_background'] for r in rows.values()),
                ambiguous_regions=sum(not r['all_anchors_in_one_background'] for r in rows.values()),
                regions=rows,pairs=pairs)


def compare_response(clean,candidate,reference=None):
    """同原生 ROI 的响应保留/新增及多锚点拓扑；没有模型更新。"""
    clean,candidate=_probability(clean),_probability(candidate)
    if clean.shape!=candidate.shape:
        raise ValueError('Reference and candidate must match native ROI shape')
    high=clean>=PROTOCOL['strong_threshold']
    low=clean<PROTOCOL['low_reference_threshold']
    difference=candidate.astype(np.float64)-clean.astype(np.float64)
    high_count=int(high.sum());low_count=int(low.sum())
    lost=int((high&(candidate<PROTOCOL['background_threshold'])).sum())
    added45=int((low&(candidate>=.45)).sum());added65=int((low&(candidate>=.65)).sum())
    return dict(pixels=int(clean.size),mean_reference=float(clean.mean(dtype=np.float64)),
                mean_candidate=float(candidate.mean(dtype=np.float64)),mae=float(np.abs(difference).mean()),
                signed_mean=float(difference.mean()),reference_high_pixels=high_count,lost_strong_pixels=lost,
                strong_to_low_fraction=_divide(lost,high_count),reference_low_pixels=low_count,
                new_response_045_pixels=added45,new_response_065_pixels=added65,
                new_response_045_fraction=_divide(added45,low_count),new_response_065_fraction=_divide(added65,low_count),
                topology=_topology(reference or build_reference_regions(clean),candidate))


def _q(values):
    values=[float(v) for v in values if v is not None]
    if not values:return dict(n=0,median=None,q25=None,q75=None)
    q=np.quantile(values,[.25,.5,.75])
    return dict(n=len(values),median=float(q[1]),q25=float(q[0]),q75=float(q[2]))


def summarize_sources(rows):
    """每源先独立统计，再给源等权分位数；同时保留合并分母。"""
    grouped=defaultdict(list)
    for row in rows:grouped[(row['domain'],row['variant'])].append(row)
    output=[]
    for (domain,variant),items in sorted(grouped.items()):
        totals={key:sum(r['metrics'][key] for r in items) for key in [
            'pixels','reference_high_pixels','lost_strong_pixels','reference_low_pixels',
            'new_response_045_pixels','new_response_065_pixels']}
        topo={key:sum(r['metrics']['topology'][key] for r in items) for key in [
            'total_pairs','resolved_pairs','merged_pairs','separate_pairs','ambiguous_pairs']}
        metrics={key:_q(r['metrics'][key] for r in items) for key in [
            'mae','signed_mean','mean_candidate','strong_to_low_fraction',
            'new_response_045_fraction','new_response_065_fraction']}
        metrics['merged_fraction_all_pairs']=_q(r['metrics']['topology']['merged_fraction_all_pairs'] for r in items)
        output.append(dict(domain=domain,variant=variant,sources=len(items),source_equal=metrics,
                           totals=totals,topology_totals=topo,
                           pooled_strong_to_low=_divide(totals['lost_strong_pixels'],totals['reference_high_pixels']),
                           pooled_low_to_045=_divide(totals['new_response_045_pixels'],totals['reference_low_pixels']),
                           pooled_low_to_065=_divide(totals['new_response_065_pixels'],totals['reference_low_pixels']),
                           pooled_merge_all_pairs=_divide(topo['merged_pairs'],topo['total_pairs'])))
    return output


def _heat(values):
    return cv2.applyColorMap(np.uint8(np.clip(values,0,1)*255),cv2.COLORMAP_VIRIDIS)


def _delta(values):
    values=np.clip(values/.5,-1,1);out=np.ones((*values.shape,3),np.float32);pos=values>=0
    out[...,0][pos]=1-values[pos];out[...,1][pos]=1-values[pos]
    out[...,1][~pos]=1+values[~pos];out[...,2][~pos]=1+values[~pos]
    return np.uint8(out*255)


def _panel(values,title):
    values=cv2.resize(values,(320,320),interpolation=cv2.INTER_AREA)
    panel=np.full((350,320,3),245,np.uint8);panel[30:]=values
    cv2.putText(panel,title,(6,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(20,20,20),1,cv2.LINE_AA)
    return panel


def write_gallery(source,domain,clean,rgb,variants,reference,rows,outdir):
    """固定全 ROI，不根据候选变化挑最好看的局部。"""
    output=[];by_name={r['variant']:r for r in rows}
    for name,probability in variants.items():
        if name=='sigma0':continue
        visual=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR) if rgb is not None else _heat(clean)
        top=np.hstack([_panel(visual,'Original RGB / fixed ROI'),_panel(_heat(clean),'Original N0 reference'),
                       _panel(_heat(probability),name),_panel(_delta(probability-clean),'Delta: blue - / red +')])
        overlay=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR) if rgb is not None else _heat(clean)
        topology=by_name[name]['metrics']['topology']
        merged_ids={v for pair in topology['pairs'] if pair['state']=='merged' for v in pair['reference_regions']}
        for region_id in merged_ids:
            contour=(reference['labels']==region_id).astype(np.uint8)
            boundaries,_=cv2.findContours(contour,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay,boundaries,-1,(0,80,255),1)
            for y,x in reference['regions'][region_id]['anchors_yx']:
                cv2.circle(overlay,(x,y),2,(0,255,255),-1)
        signal=np.zeros((*clean.shape,3),np.uint8)
        signal[(clean>=.65)&(probability<.45)]=(255,180,0)
        signal[(clean<.2)&(probability>=.45)]=(0,80,255)
        bottom=np.hstack([_panel(overlay,'Merged reference regions / anchors'),
                          _panel(signal,'Blue: strong lost; red: low raised'),
                          _panel(np.repeat(np.uint8(clean<.45)[...,None]*255,3,axis=2),'Reference background < .45'),
                          _panel(np.repeat(np.uint8(probability<.45)[...,None]*255,3,axis=2),'Candidate background < .45')])
        header=np.full((52,top.shape[1],3),245,np.uint8)
        cv2.putText(header,f'{source} | {domain} | {name} | entire preregistered native ROI',(8,21),cv2.FONT_HERSHEY_SIMPLEX,.55,(20,20,20),1,cv2.LINE_AA)
        cv2.putText(header,'N0-relative signal topology only; no GT. Probability fixed [0,1], delta fixed [-.5,.5].',(8,43),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,20,20),1,cv2.LINE_AA)
        stem=re.sub(r'[^A-Za-z0-9_.-]+','_',f'{domain}_{source}_{name}')
        path=outdir/f'{stem}.jpg';cv2.imwrite(str(path),np.vstack([header,top,bottom]),[cv2.IMWRITE_JPEG_QUALITY,93])
        output.append(str(path))
    return output


def _load_captures(manifest_path,expected_sources):
    manifest_path=Path(manifest_path)
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest.get('format')!='boundary_rgb_probe_v1' or manifest.get('complete') is not True
            or manifest.get('zero_gate',{}).get('all_exact') is not True):
        raise ValueError('Completed RGB capture and exact sigma0 bridge required before analysis')
    groups=defaultdict(dict)
    for row in manifest['files']:
        source,domain=row['source'],row['domain']
        sigma=float(row['sigma_native'])
        if sigma not in (0.,2.,4.):
            raise ValueError('Only preregistered native sigma 0/2/4 are supported')
        mode=f'sigma{int(sigma)}'
        if mode in groups[(domain,source)]:
            raise ValueError('Duplicate source/domain/sigma capture')
        path=Path(row['path'])
        if not path.is_absolute():path=manifest_path.parent/path
        if _sha(path)!=row['sha256']:
            raise ValueError(f'Capture file SHA mismatch: {source}/{mode}')
        with np.load(path,allow_pickle=False) as payload:
            if {'instance_map','truth_boundary','gt','target_gt'}&set(payload.files):
                raise ValueError('RGB diagnosis must not load GT-bearing caches')
            probability=_probability(payload['boundary512']).copy()
            rgb=payload['rgb512'].copy();original=payload['reference_rgb512'].copy()
        if (rgb.shape!=(*probability.shape,3) or original.shape!=rgb.shape
                or rgb.dtype!=np.uint8 or original.dtype!=np.uint8):
            raise ValueError('Capture RGB must be native HWC uint8 aligned with probability')
        groups[(domain,source)][mode]=dict(probability=probability,rgb=rgb,original=original,receipt=row)
    if len(groups)!=expected_sources:
        raise ValueError(f'Expected {expected_sources} sources, received {len(groups)}')
    for key,views in groups.items():
        if set(views)!={'sigma0','sigma2','sigma4'}:
            raise ValueError(f'Missing preregistered blur levels: {key}')
        original=views['sigma0']['original']
        if not np.array_equal(views['sigma0']['rgb'],original):
            raise ValueError('Sigma0 RGB is not the unchanged original')
        for view in views.values():
            if not np.array_equal(view['original'],original):
                raise ValueError('Source content/alignment differs across blur levels')
            if view['receipt']['roi_box']!=views['sigma0']['receipt']['roi_box']:
                raise ValueError('Native ROI differs across blur levels')
    return manifest,groups


def _load_extra(extra_path,groups):
    """可选：已完成的旧补边模型交叉前向；此工具本身不占 GPU、不执行模型。"""
    if extra_path is None:return None,{}
    path=Path(extra_path);manifest=json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('format')!='boundary_rgb_restorer_pairs_v1' or manifest.get('complete') is not True:
        raise ValueError('Additional paired predictions must be complete')
    output=defaultdict(dict)
    for row in manifest['files']:
        identity=(row['domain'],row['source'])
        if identity not in groups:raise ValueError('Extra prediction has an unregistered source')
        sigma=float(row['sigma_native']);base=f'sigma{int(sigma)}'
        if base not in groups[identity] or sigma not in (0.,2.,4.):
            raise ValueError('Extra prediction has an unregistered blur level')
        if row['input_capture_sha256']!=groups[identity][base]['receipt']['sha256']:
            raise ValueError('Restorer input must bind exact captured probability cache')
        name=base+'_'+row['model']
        if name in output[identity]:raise ValueError('Duplicate paired model output')
        cache=Path(row['path'])
        if not cache.is_absolute():cache=path.parent/cache
        if _sha(cache)!=row['sha256']:raise ValueError('Extra model cache SHA mismatch')
        with np.load(cache,allow_pickle=False) as payload:probability=_probability(payload[row.get('key','output')]).copy()
        if probability.shape!=groups[identity][base]['probability'].shape:
            raise ValueError('Extra prediction shape mismatch')
        output[identity][name]=dict(probability=probability,sigma=sigma,receipt=row)
    return manifest,output


def _write_report(rows,summaries,path):
    source_count=len({(r['domain'],r['source']) for r in rows})
    lines=['# 原生 RGB 模糊引起的 N0 响应变化','',
           f'固定{source_count}个训练源中心ROI，Gaussian sigma=0/2/4原生像素；同源固定窗口、噪声与拼接。sigma0先与旧缓存逐值核对。无GT、无测试图训练；下列数字均以原N0响应为参考，不是真实分割准确率。','',
           '背景连通使用固定阈值0.45与8邻接，沿用已有响应探针口径。仅比较原图中不触ROI边缘且有至少3个空间分散、深内部低响应锚点的闭合区域；邻接对由原边带的最近标签相邻关系确定。只有两区各自全部锚点仍在同一背景分量、且两区该分量相同时才记合并，其余不确定情况单列。没有可用邻接对时分母为0、比例为null，不能解释为0%粘连。','',
           '| 源 / 域 | 档位 | MAE | 原≥.65跌至<.45 | 原<.2升至≥.45 | 原<.2升至≥.65 | 原邻接对 | 合并 | 不确定 |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    def percentage(value):return 'NA' if value is None else f'{100*value:.2f}%'
    for row in rows:
        if row['variant']=='sigma0':continue
        m=row['metrics'];t=m['topology']
        lines.append(f"| {row['source']} / {row['domain']} | {row['variant']} | {m['mae']:.5f} | {percentage(m['strong_to_low_fraction'])} ({m['lost_strong_pixels']}/{m['reference_high_pixels']}) | {percentage(m['new_response_045_fraction'])} | {percentage(m['new_response_065_fraction'])} | {t['total_pairs']} | {t['merged_pairs']} | {t['ambiguous_pairs']} |")
    lines+=['',f'各源等权中位数/IQR与所有计数分母保留在JSON。区域对是诊断单位，不是独立统计样本；{source_count}个中心ROI不能代表完整1000图或比赛测试分布。',
            '', '若包含旧补边模型输出，仅为同一512配对上的交叉检查，没有重跑全图上下文或最终实例读出，不能解释为部署收益。',
            '', '不以单个中心点超过阈值作为补口成功；新增或消失响应也不自动等于错误，原N0本身可能错误。']
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run_analysis(manifest_path,output_dir,*,expected_sources=8,extra_manifest=None):
    manifest,groups=_load_captures(manifest_path,int(expected_sources))
    extras,additional=_load_extra(extra_manifest,groups)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);gallery=out/'gallery';gallery.mkdir(exist_ok=True)
    rows,source_rows,galleries=[],[],[]
    for (domain,source),views in sorted(groups.items()):
        clean=views['sigma0']['probability'];reference=build_reference_regions(clean)
        variants={name:views[name]['probability'] for name in ('sigma0','sigma2','sigma4')}
        variants.update({name:item['probability'] for name,item in additional.get((domain,source),{}).items()})
        these=[]
        for name,probability in variants.items():
            row=dict(source=source,domain=domain,variant=name,
                     sigma_native=float(name.split('_')[0].removeprefix('sigma')),
                     metrics=compare_response(clean,probability,reference))
            these.append(row);rows.append(row)
        baseline=these[0]['metrics']
        if baseline['mae']!=0 or baseline['topology']['merged_pairs']!=0 or baseline['topology']['ambiguous_pairs']!=0:
            raise RuntimeError('Identity topology baseline must reproduce the reference exactly')
        source_rows.append(dict(source=source,domain=domain,reference=reference['summary'],
                                reliable_regions=reference['regions'],adjacent_pairs=reference['pairs'],
                                rejected_regions=reference['rejected_regions']))
        galleries.extend(write_gallery(source,domain,clean,views['sigma0']['original'],variants,reference,these,gallery))
    summaries=summarize_sources(rows)
    result=dict(format='boundary_rgb_analysis_v1',complete=True,trained=False,gpu_inference=False,
                target_not_gt=True,not_final_instance_readout=True,protocol=PROTOCOL,
                capture_manifest=str(Path(manifest_path).resolve()),capture_manifest_sha256=_sha(manifest_path),
                extra_manifest_sha256=_sha(extra_manifest) if extra_manifest else None,
                source_count=len(groups),source_rows=source_rows,rows=rows,summary=summaries,gallery=galleries,
                model_provenance=extras.get('models') if extras else None,source_sha256=_sha(__file__))
    _write(out/'analysis.json',result)
    _write_report(rows,summaries,out/'analysis.md')
    with (out/'source_table.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.writer(file);writer.writerow(['source','domain','variant','mae','strong_to_low','low_to_045','low_to_065','pairs','merged','ambiguous'])
        for row in rows:
            m=row['metrics'];t=m['topology'];writer.writerow([row['source'],row['domain'],row['variant'],m['mae'],m['strong_to_low_fraction'],m['new_response_045_fraction'],m['new_response_065_fraction'],t['total_pairs'],t['merged_pairs'],t['ambiguous_pairs']])
    print(json.dumps(dict(complete=True,source_count=len(groups),rows=len(rows),output=str(out/'analysis.json')),ensure_ascii=False))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--expected-sources',type=int,default=8)
    parser.add_argument('--extra-manifest')
    args=parser.parse_args()
    run_analysis(args.manifest,args.output,expected_sources=args.expected_sources,extra_manifest=args.extra_manifest)


if __name__=='__main__':main()
