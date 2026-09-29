# -*- coding: utf-8 -*-
"""双尺度训练完成检查：全测试输出差异和既有四图原标域诊断，均不作官方精度。"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.mim_dataset import list_images
from tools.analyze_affinity_connectivity import load_directory, read, relation, render, rows, write
from utils.config import load_config, project_path
from utils.patch_diagnostic import phase_description, trusted_phase_metrics


def folders(config, root):
    return {
        'control_global': Path(project_path(config, config['affinity_native']['reused_control'], 'deployment')),
        'control_1024': root/'control_views/patch1024',
        'p1024_global': root/'p1024/deployment/global',
        'p1024_native': root/'p1024/deployment/patch1024',
        'control_512': root/'control_views/patch512',
        'p512_global': root/'p512/deployment/global',
        'p512_native': root/'p512/deployment/patch512',
    }


COMPARISONS = [('control_global', 'p1024_global'), ('control_global', 'p512_global'),
               ('control_1024', 'p1024_native'), ('control_512', 'p512_native'),
               ('control_global', 'p1024_native'), ('control_global', 'p512_native')]


def quantiles(values):
    return dict(zip(('p10', 'median', 'p90'), map(float, np.quantile(values, [.1, .5, .9])))) if values else None


def audit(config, root):
    from train_backend_adaptation import sha
    from tools.run_affinity_native import verify_run
    state = read(root/'pipeline_status.json')
    if state['status'] != 'complete':
        raise RuntimeError('Pipeline incomplete')
    checked = verify_run(root, 1280)
    changed = [name for name, digest in state['sources'].items() if sha(ROOT/name) != digest]
    if changed:
        raise RuntimeError('Runtime source changed since training: '+str(changed))
    control = Path(project_path(config, config['affinity_native']['reused_control']))
    if sha(control/'final.pt') != config['affinity_native']['control_sha256']:
        raise RuntimeError('Control checkpoint differs')
    receipts = rows(control/'steps.jsonl')
    checked['training'] = {}
    for size in (1024, 512):
        folder = root/f'p{size}'
        status = read(folder/'status.json')
        steps = rows(folder/'steps.jsonl')
        epochs = rows(folder/'epochs.jsonl')
        if sha(folder/'final.pt') != status['final_checkpoint_sha256']:
            raise RuntimeError('Final checkpoint differs')
        for old, new in zip(receipts, steps):
            for key in ('epoch', 'step', 'seed', 'manual', 'pseudo', 'manual_draw', 'pseudo_draw'):
                if old[key] != new[key]:
                    raise RuntimeError('Unpaired draw: '+key)
        if len(epochs) != 20 or epochs[-1]['updates'] != 1280:
            raise RuntimeError('Incomplete epochs')
        expected = [folder/'monitor'/f'epoch_{e:03d}'/(name+'_'+suffix+'.png')
                    for e in (0, 1, 5, 10, 15, 20)
                    for name in config['backend_adaptation']['monitor']['images']
                    for suffix in ('full', 'detail', 'boundary')]
        if not all(path.is_file() for path in expected):
            raise RuntimeError('Missing process image')
        checked['training'][f'p{size}'] = dict(status=status, first=epochs[0], last=epochs[-1],
            paired_receipts=len(steps), process_images=len(expected),
            monitor_browser_backups=len(list((folder/'monitor').rglob('*-checkpoint.png'))),
            last5={k:float(np.mean([r[k] for r in epochs[-5:]])) for k in ('manual_bce', 'pseudo_bce')},
            previous5={k:float(np.mean([r[k] for r in epochs[-10:-5]])) for k in ('manual_bce', 'pseudo_bce')},
            crop_retries=sum(sum(r['crop_attempts'][0])+sum(r['crop_attempts'][1]) for r in steps))
    return checked


def training_plot(root, output, config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), layout='constrained')
    control = Path(project_path(config, config['affinity_native']['reused_control']))
    for name, folder in [('control', control), ('p1024', root/'p1024'), ('p512', root/'p512')]:
        values = rows(folder/'epochs.jsonl')
        for ax, key in zip(axes, ('manual_bce', 'pseudo_bce', 'next_lr')):
            ax.plot([r['epoch'] for r in values], [r[key] for r in values], label=name)
            ax.set_title(key); ax.set_xlabel('Epoch'); ax.grid(alpha=.2)
    axes[0].legend()
    fig.suptitle('Different crop supervision: loss is not comparable accuracy')
    fig.savefig(output/'training.png', dpi=150); plt.close(fig)


def analyze_test(config, root, output):
    result = dict(scope='Unlabeled output differences, not accuracy or official area score',
                  audit=audit(config, root), images=[], selected={})
    training_plot(root, output, config)
    paths = [Path(p) for p in list_images(project_path(config, config['inference']['test_dir']))]
    if len(paths) != 100:
        raise RuntimeError('Expected exactly 100 test images')
    directories = folders(config, root)
    fixed = config['backend_adaptation']['monitor']['images']
    others = [p.stem for p in paths if p.stem not in fixed]
    random = sorted(np.random.default_rng(20260929).choice(others, 6, replace=False).tolist())
    selected = set(random) | {'test_101', 'test_116'}
    result['selected'] = dict(fixed=['test_101', 'test_116'], random=random, seed=20260929)
    for index, path in enumerate(paths):
        image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        predictions = {k:load_directory(folder, path.stem) for k, folder in directories.items()}
        if any(ids.shape != image.shape[:2] or any(c not in (0, 1) for c in classes.values())
               for ids, classes in predictions.values()):
            raise RuntimeError('Invalid final output: '+path.stem)
        base, bc = predictions['control_global']
        areas = np.bincount(base.ravel())
        fa = [areas[int(k)] for k,c in bc.items() if c == 1]
        reference = float(np.median(fa)) if fa else None
        record = dict(source=path.stem, views={}, relations={})
        for key, (ids, classes) in predictions.items():
            stats = phase_description(ids, classes)
            areas = np.bincount(ids.ravel())
            fa = np.array([areas[int(k)] for k,c in classes.items() if c == 1])
            stats['ferrite'].update(control_median_area=reference,
                under_quarter_control_median=int((fa < reference*.25).sum()) if reference else None)
            record['views'][key] = stats
        # 只在相同裁窗尺度内计算分裂/合并关系；这是变化而不是正确率。
        for left, right in COMPARISONS[2:4]:
            a, ac = predictions[left]; b, bcl = predictions[right]
            record['relations'][left+'__'+right] = relation(a, b, ac, bcl)
        result['images'].append(record)
        if path.stem in selected:
            for size in (1024, 512):
                keys = ['control_global', f'control_{size}', f'p{size}_native']
                names = ['control whole', f'control native{size}', f'trained native{size}']
                for suffix, crop in [('full',None), ('detail',[.25,.25,.75,.75])]:
                    render(path, [predictions[k] for k in keys], names,
                           output/(path.stem+f'_{size}_{suffix}.png'), path.stem, crop=crop, width=400)
            keys = ['control_global', 'p1024_native', 'p512_native']
            render(path, [predictions[k] for k in keys], ['control whole','trained native1024','trained native512'],
                   output/(path.stem+'_scales.png'), path.stem, width=400)
        if index % 10 == 0:
            print('test', index+1, path.stem, flush=True)
    result['totals'] = {}
    for key in directories:
        result['totals'][key] = {phase:{k:sum(r['views'][key][phase][k] for r in result['images'])
            for k in ('instances','pixels',*(['tiny_under200','under_quarter_control_median'] if phase=='ferrite' else []))}
            for phase in ('ferrite','pearlite')}
    result['comparisons'] = {}
    for left, right in COMPARISONS:
        changes = {}
        for key in ('instances','pixels','mean_area'):
            ratios = [r['views'][right]['ferrite'][key]/r['views'][left]['ferrite'][key]-1
                      for r in result['images'] if r['views'][left]['ferrite'][key] and r['views'][right]['ferrite'][key]]
            changes['ferrite_'+key+'_relative'] = quantiles(ratios)
        if left+'__'+right in result['images'][0]['relations']:
            values = [r['relations'][left+'__'+right] for r in result['images']]
            changes['relations'] = {k:sum(r[k] for r in values) for k in ('split_relations','merge_relations',
                'matched_iou50','class_changed_matched_iou50','class_changed_pixels','common_covered_pixels')}
        result['comparisons'][left+'__'+right] = changes
    write(output/'test.json', result)
    print(json.dumps({k:result[k] for k in ('totals','comparisons','selected')},indent=2), flush=True)


def analyze_gt(config, root, output):
    import torch
    from models.backend_adaptation import load_backend
    from tools.affinity_native_views import predict_views
    from tools.probe_affinity_patch import load_gt
    from train_affinity_connectivity import get_restorer
    from train_backend_adaptation import sha
    prior = Path(project_path(config, config['affinity_native']['prior_probe']))
    cases = [c for c in read(prior/'cases.json') if c['kind']=='train']
    if len(cases) != 4:
        raise RuntimeError('Expected prior four training sources')
    records = []
    for case in cases:
        gt, trust, gc = load_gt(case, config)
        for key, variant in [('control_global','global_d5a'), ('control_1024','patch1024_d5a'), ('control_512','patch512_d5a')]:
            ids, classes = load_directory(prior/case['name']/variant, case['name'])
            records.append(dict(source=case['name'],view=key,metrics=trusted_phase_metrics(gt,gc,trust,ids,classes)))
    restorer = get_restorer(config, 'cuda')
    for size in (1024, 512):
        checkpoint = root/f'p{size}/final.pt'
        model, _ = load_backend(checkpoint, config, 'cuda')
        model.eval().requires_grad_(False)
        for case in cases:
            gt, trust, gc = load_gt(case, config)
            _, views = predict_views(model, restorer, case['path'], config, 'cuda', case['seed'], [0, size])
            for view, values in views.items():
                key = f'p{size}_'+('global' if view=='global' else 'native')
                ids, classes = values['instances'], values['classes']
                directory = output/'train'/case['name']/key
                directory.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(directory/(case['name']+'_inst.png')),ids):
                    raise RuntimeError('Could not save PNG')
                write(directory/(case['name']+'_class.json'), classes)
                records.append(dict(source=case['name'],view=key,checkpoint_sha256=sha(checkpoint),
                                    metrics=trusted_phase_metrics(gt,gc,trust,ids,classes)))
            print('GT',size,case['name'],flush=True)
        del model
        torch.cuda.empty_cache()
    totals = {}
    for key in folders(config,root):
        selected = [r for r in records if r['view']==key]
        if len(selected)!=4:
            raise RuntimeError('Missing GT comparison')
        totals[key] = {}
        for phase in ('ferrite','pearlite'):
            values = [r['metrics'][phase] for r in selected]
            sums = {k:sum(r[k] for r in values) for k in ('gt_instances','pred_instances_in_known_domain',
                    'matches','iou_sum','split_gt','merged_pred')}
            sums.update(matched_iou=sums['iou_sum']/max(1,sums['matches']),
                        gt_penalized_iou=sums['iou_sum']/max(1,sums['gt_instances']))
            totals[key][phase] = sums
    for case in cases:
        gt, trust, gc = load_gt(case,config)
        predictions = [(gt.astype(np.uint16), {str(k):v for k,v in gc.items()})]
        predictions += [load_directory(prior/case['name']/'global_d5a',case['name'])]
        predictions += [load_directory(output/'train'/case['name']/key,case['name'])
                        for key in ('p1024_native','p512_native')]
        render(case['path'], predictions,['GT (has omissions)','control whole','trained native1024','trained native512'],
               output/(case['name']+'_detail.png'),case['name']+' known training source',crop=[.25,.25,.75,.75],width=400)
    report = dict(scope='Seen training sources, original covered GT only; unknown masked on both sides. Not held-out or official accuracy.',
                  cases=cases,rows=records,totals=totals)
    write(output/'gt.json',report)
    print(json.dumps(totals,indent=2),flush=True)


def make_index(output):
    items = ['<!doctype html><meta charset="utf-8"><title>原生裁块训练分析</title>',
             '<style>body{font:16px system-ui;margin:24px;background:#eee}img{max-width:100%;background:white}section{margin:30px 0}a{color:#247}</style>',
             '<h1>1024 / 512 原生裁块训练对照</h1>',
             '<p>金色=铁素体，蓝色=珠光体。边界变化不等于测试精度提升；训练GT图存在缺标。</p>',
             '<p>固定样本test101、116；其余六图用种子20260929从非monitor图中预先随机抽取。</p>',
             '<p><a href="test.json">100图汇总及逐图数据</a> · <a href="gt.json">4张已见GT诊断</a></p>']
    for path in [output/'training.png', *sorted(output.glob('test*_scales.png')),
                 *sorted(output.glob('test*_detail.png')), *sorted(output.glob('train*_detail.png'))]:
        if path.exists():
            name=html.escape(path.name)
            items.append(f'<section><h2>{name}</h2><a href="{name}"><img loading="lazy" src="{name}"></a></section>')
    (output/'index.html').write_text('\n'.join(items),encoding='utf8')


def render_extremes(config, root, output):
    """按已公布的面积变化排序补看极端样例，不把它们混入随机样本。"""
    report=read(output/'test.json')
    chosen={}
    for key in ('p1024_native','p512_native'):
        valid=[r for r in report['images'] if r['views']['control_global']['ferrite']['mean_area']
               and r['views'][key]['ferrite']['mean_area']]
        ordered=sorted(valid,key=lambda r:r['views'][key]['ferrite']['mean_area']/r['views']['control_global']['ferrite']['mean_area'])
        for side,row in [('min',ordered[0]),('max',ordered[-1])]:
            chosen.setdefault(row['source'],[]).append(key+'_'+side)
    directories=folders(config,root)
    paths={Path(p).stem:Path(p) for p in list_images(project_path(config,config['inference']['test_dir']))}
    for source,reasons in chosen.items():
        predictions=[load_directory(directories[key],source) for key in ('control_global','p1024_native','p512_native')]
        render(paths[source],predictions,['control whole','trained native1024','trained native512'],
               output/(source+'_extreme_detail.png'),source+' area-change extreme',crop=[.25,.25,.75,.75],width=400)
    write(output/'extreme_selection.json',dict(scope='Selected by area-change extremes, not representative random samples',images=chosen))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',default='config/train/affinity_native.yaml')
    parser.add_argument('--output',default='outputs/native_analysis')
    parser.add_argument('--mode',choices=['test','gt','index','extremes','all'],default='all')
    args=parser.parse_args()
    config=load_config(args.config)
    root=Path(project_path(config,config['backend_adaptation']['output_dir']))
    output=Path(project_path(config,args.output));output.mkdir(parents=True,exist_ok=True)
    if args.mode in ('test','all'):analyze_test(config,root,output)
    if args.mode in ('gt','all'):analyze_gt(config,root,output)
    if args.mode in ('extremes','all'):render_extremes(config,root,output)
    make_index(output)
