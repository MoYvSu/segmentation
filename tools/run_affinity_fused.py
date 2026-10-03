# -*- coding: utf-8 -*-
"""先新GT门禁及8步短测，再独立20轮融合监督；不重训控制、不自动提交。"""
from __future__ import annotations

import argparse
import ast
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.probe_affinity_spatial import compare_spatial_boundaries
from tools.run_affinity_short import precision_record, verify_package
from tools.run_affinity_sweep import audit_control, runtime_contract
from tools.verify_affinity_newgt import audit_new_gt
from train_affinity_fused import CONFIG_PATH, REFERENCE, REFERENCE_SHA256, reference_config, verify_reference, verify_run
from train_backend_adaptation import fusion_options, sha, tensor_digest, write_json
from utils.config import load_config, project_path

# 初次8步已通过，原门禁误将YAML数值键与JSON文字键直接比较。
# 保留原完整技术回执；修复后的队列使用独立目录，不改写旧快照。
SMOKE = ROOT/'outputs/affinity_fused_smoke2'
FORMAL = ROOT/'outputs/affinity_fused'
NEW_FILES = (CONFIG_PATH, 'train_affinity_fused.py', 'tools/run_affinity_fused.py',
    'utils/affinity_fused_loss.py', 'tools/verify_affinity_newgt.py', 'tools/probe_affinity_spatial.py')


def source_inventory(parent):
    found, pending = set(parent), list(NEW_FILES)
    while pending:
        name = pending.pop()
        if name in found:
            continue
        path = ROOT/name
        if not path.is_file():
            raise RuntimeError('Missing source: '+name)
        found.add(name)
        if path.suffix in ('.yaml','.yml'):
            base = (yaml.safe_load(path.read_text(encoding='utf8')) or {}).get('_base')
            if base:
                pending.append((path.parent/base).resolve().relative_to(ROOT).as_posix())
            continue
        if path.suffix != '.py':
            raise RuntimeError('Unsupported source type')
        package, modules = Path(name).parent.parts, []
        for node in ast.walk(ast.parse(path.read_text(encoding='utf8'))):
            if isinstance(node, ast.Import):
                modules.extend(a.name.split('.') for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                prefix = list(package[:len(package)-node.level+1]) if node.level else []
                module = prefix+(node.module.split('.') if node.module else [])
                modules.append(module)
                modules.extend(module+a.name.split('.') for a in node.names if a.name != '*')
        for parts in modules:
            if not parts:
                continue
            for item in (Path(*parts).with_suffix('.py'), Path(*parts)/'__init__.py'):
                if (ROOT/item).is_file():
                    pending.append(item.as_posix())
            for n in range(1,len(parts)):
                item = Path(*parts[:n])/'__init__.py'
                if (ROOT/item).is_file():
                    pending.append(item.as_posix())
    return {name:sha(ROOT/name) for name in sorted(found)}


def verify_snapshot(root, sources):
    for name, digest in sources.items():
        if (Path(name).is_absolute() or '..' in Path(name).parts or sha(ROOT/name) != digest
                or sha(Path(root)/'source'/name) != digest):
            raise RuntimeError('Protected source or snapshot changed: '+name)


def parent_contract():
    """直接复核封存源码和环境；不递归重放与本候选无关的历史扫参。"""
    folder = ROOT/'outputs/affinity_p100'
    receipt = json.loads((folder/'pipeline_status.json').read_text())
    if (receipt.get('status') != 'complete' or receipt.get('stage') != 'done'
            or receipt.get('control_retrained') is not False or len(receipt.get('sources',{})) != 134):
        raise RuntimeError('The registered completed 134-source parent seal is required')
    verify_snapshot(folder,receipt['sources'])
    return dict(sources=receipt['sources'],runtime=receipt['runtime'],
                pipeline_sha256=sha(folder/'pipeline_status.json'))


def verify_newgt(config, receipt):
    if (receipt['npz_cohort_sha256'] != config['affinity_fused']['gt_npz_cohort_sha256']
            or receipt['class_cohort_sha256'] != config['affinity_fused']['gt_class_cohort_sha256']
            or receipt.get('original_covered_is_supervision_mask') is not False):
        raise RuntimeError('Processed newGT identity/support differs')


def spatial_probe(config, root, *, after=False):
    """固定4个真实训练draw：新GT全支持，逐图gated/short或旧/新gated同输入比较。"""
    from data.affinity_mixed import build_datasets
    from models.backend_adaptation import load_backend, restore_first
    from torch.utils.data import default_collate
    from train_affinity_connectivity import get_restorer
    from train_affinity_connectivity import draw_receipt
    from train_direct_semantic_affinity import move_batch, set_seed
    from utils.affinity_fusion import affinity_boundary_probability

    config_path = Path(project_path(config, REFERENCE))
    reference_rows = [json.loads(s) for s in (config_path/'steps.jsonl').read_text().splitlines()]
    selected, counts = [], {'whole':0,'native1024':0}
    for row in reference_rows[:64]:
        view = row['training_view']
        if counts[view] < 2:
            selected.append(row);counts[view] += 1
    if len(selected) != 4:
        raise RuntimeError('Two fixed whole and two fixed native1024 draws are required')
    manual, _, _ = build_datasets(config,1024)
    manual.set_epoch(1)
    checkpoint = root/'candidate/final.pt' if after else config_path/'final.pt'
    model, _ = load_backend(checkpoint,config,'cuda')
    restorer = get_restorer(config,'cuda')
    before = tensor_digest(model.state_dict().items())
    restorer_before = tensor_digest(restorer.state_dict().items())
    _, kwargs = fusion_options(config)
    folder = root/('spatial_after' if after else 'spatial_before')
    folder.mkdir(exist_ok=False)
    records = []
    with torch.no_grad():
        for row in selected:
            index = int(row['manual_draw']['draw_index'][0])
            raw = default_collate([manual[index]])
            if (raw['name'] != row['manual'] or raw['training_view'][0] != row['training_view']
                    or raw['crop_box'].tolist() != row['manual_crop'] or draw_receipt(raw) != row['manual_draw']):
                raise RuntimeError('Fixed spatial source/view identity changed')
            batch = move_batch(raw,'cuda')
            set_seed(row['seed'])
            restored = restore_first(restorer,batch['image'],row['seed']+1000000)
            restored_sha = tensor_digest([('restored',restored)])
            logits = model.affinity_decoder(model.encoder(restored))['affinity_logits'].float()
            ids = batch['affinity_instance_map'][0].cpu().numpy()
            content = batch['affinity_valid_content'][0,0].cpu().numpy().astype(bool)
            trusted = content & (ids > 0)
            gated = affinity_boundary_probability(logits,mode='gated',**kwargs)[0,0].cpu().numpy()
            if after:
                previous = root/'spatial_before'/f'draw_{index:03d}.npz'
                with np.load(previous,allow_pickle=False) as saved:
                    if (not np.array_equal(saved['ids'],ids) or not np.array_equal(saved['content'],content)
                            or saved['input_sha256'].item() != tensor_digest([('input',batch['image'])])
                            or saved['restored_sha256'].item() != restored_sha):
                        raise RuntimeError('Before/after spatial newGT/input identity differs')
                    arms = {'control_gated':saved['gated'].copy(),'candidate_gated':gated}
            else:
                short = affinity_boundary_probability(logits,mode='short',**kwargs)[0,0].cpu().numpy()
                arms = {'gated':gated,'short':short}
                np.savez_compressed(folder/f'draw_{index:03d}.npz',ids=ids,content=content,gated=gated,
                    short=short,input_sha256=tensor_digest([('input',batch['image'])]),restored_sha256=restored_sha)
            report = compare_spatial_boundaries(arms,ids,trusted,valid_content=content,
                reference='control_gated' if after else 'gated',
                threshold=config['inference']['boundary_threshold'],return_maps=False)['report']
            records.append(dict(draw=index,source=row['manual'],view=row['training_view'],seed=row['seed'],report=report))
            print('spatial', 'after' if after else 'before',index,flush=True)
    if (tensor_digest(model.state_dict().items()) != before
            or tensor_digest(restorer.state_dict().items()) != restorer_before):
        raise RuntimeError('Spatial diagnostic changed frozen model/restorer state')
    write_json(folder/'report.json',dict(complete=True,checkpoint_sha256=sha(checkpoint),
        support='processed newGT known IDs including filled pixels; unknown/padding ignored',
        scope='four seen training draws; no test GT or independent accuracy',rows=records))
    del model,restorer
    gc.collect();torch.cuda.empty_cache()


def render_final(config, root):
    from data.mim_dataset import list_images
    from tools.analyze_affinity_connectivity import load_directory, render
    control = Path(project_path(config,REFERENCE,'deployment/patch1024'))
    candidate = root/'candidate/deployment/patch1024'
    manifest = json.loads((candidate.parent/'manifest.json').read_text())
    rows = manifest.get('images',[])
    names = [row['source'] for row in rows]
    files = list(map(Path,list_images(project_path(config,config['inference']['test_dir']))))
    wanted = {stem+suffix for stem in names for suffix in ('_inst.png','_class.json')}
    if (manifest.get('complete') is not True or manifest.get('epoch') != 20
            or manifest.get('sizes') != [1024] or manifest.get('checkpoint_sha256') != sha(root/'candidate/final.pt')
            or len(names) != 100 or len(set(names)) != 100 or set(names) != {p.stem for p in files}
            or any(row.get('view') != 'patch1024' for row in rows)
            or wanted != {p.name for p in candidate.iterdir() if p.is_file()}):
        raise RuntimeError('Final native1024/e20/100-source/200-file output contract differs')
    gallery = root/'gallery';gallery.mkdir()
    pages = ['<!doctype html><meta charset="utf-8"><title>Fused ranking</title>',
        '<p>Fixed mixed balance control vs newGT fused ranking e20. Prediction only, no test GT.</p>']
    for path in files:
        render(path,[load_directory(control,path.stem),load_directory(candidate,path.stem)],
            ['mixed balance e20','newGT fused ranking e20'],gallery/(path.stem+'.png'),path.stem,width=360)
        pages.append(f'<p>{path.stem}<br><img loading="lazy" width="720" src="gallery/{path.stem}.png"></p>')
    (root/'index.html').write_text('\n'.join(pages),encoding='utf8')


def run(*, smoke):
    if not torch.cuda.is_available():
        raise RuntimeError('Use GPU server sam2_env')
    torch.set_num_threads(4);cv2.setNumThreads(2)
    config = load_config(CONFIG_PATH)
    reference_config(config);verify_reference(config)
    checks = parent_contract()
    runtime = runtime_contract(config)
    if runtime != checks['runtime']:
        raise RuntimeError('Runtime differs from the existing sealed control')
    precision = precision_record(torch)
    gt = audit_new_gt(config);verify_newgt(config,gt)
    from data.mim_dataset import list_images
    files = list(map(Path,list_images(project_path(config,config['inference']['test_dir']))))
    reference_folder = Path(project_path(config,REFERENCE))
    package = verify_package(reference_folder/'deployment/patch1024',
        ROOT/'outputs/affinity_balance/mixedbalance1024.zip',ROOT/'outputs/affinity_balance/mixed_package.json',
        files,json.loads((reference_folder/'deployment/manifest.json').read_text()))
    sources = source_inventory(checks['sources'])
    root = SMOKE if smoke else FORMAL
    if root.exists() or shutil.disk_usage(ROOT).free < 6*1024**3:
        raise RuntimeError('Use a fresh output root with at least 6GiB free fast-disk space')
    if not smoke:
        gate = json.loads((SMOKE/'pipeline_status.json').read_text())
        if (gate['status'] != 'complete' or gate['config'] != json.loads(json.dumps(config)) or gate['sources'] != sources
                or gate['runtime'] != runtime or gate['newgt'] != gt or gate['precision'] != precision):
            raise RuntimeError('The exact complete 8-update newGT smoke is required')
        verify_snapshot(SMOKE,sources);verify_run(config,SMOKE/'candidate',8)
    root.mkdir(parents=True)
    for name in sources:
        target = root/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    status = dict(status='running',stage='preflight',pid=os.getpid(),smoke=smoke,config=config,
        sources=sources,runtime=runtime,precision=precision,newgt=gt,reused_control_package=package,
        completed=[],audits={},
        control_retrained=False,competition_submission=False,automatic_package=False,
        parent_pipeline_sha256=checks['pipeline_sha256'])
    def save():
        write_json(root/'pipeline_status.json',status)
    def execute(stage,command):
        status['stage']=stage;save()
        with (root/(stage+'.log')).open('x',encoding='utf8') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            status['child_pid']=child.pid;save()
            if child.wait():
                raise RuntimeError(stage+' failed; inspect its log')
        status['completed'].append(stage);status.pop('child_pid',None);save()
    save()
    try:
        if smoke:
            status['stage']='parity';save();audit_control(root)
            status['completed'].append('parity')
            status['stage']='spatial_before';save();spatial_probe(config,root)
            status['completed'].append('spatial_before')
        else:
            shutil.copytree(SMOKE/'spatial_before',root/'spatial_before')
        execute('training',[sys.executable,'-u','train_affinity_fused.py','--config',CONFIG_PATH,
            '--output',str(root/'candidate')]+(['--smoke'] if smoke else []))
        status['audits']['candidate']=verify_run(config,root/'candidate',8 if smoke else 1280)
        if not smoke:
            execute('inference',[sys.executable,'-u','train_affinity_native.py','--config',CONFIG_PATH,
                '--mode','infer','--checkpoint',str(root/'candidate/final.pt'),
                '--output',str(root/'candidate/deployment'),'--views','1024'])
            status['stage']='spatial_after';save();spatial_probe(config,root,after=True)
            status['completed'].append('spatial_after')
            status['stage']='render';save();render_final(config,root)
            status['completed'].append('render')
        verify_snapshot(root,sources)
        if audit_new_gt(config) != gt or runtime_contract(config) != runtime or precision_record(torch) != precision:
            raise RuntimeError('NewGT/source/runtime/precision changed during the experiment')
        status.update(status='complete',stage='done')
    except BaseException as exc:
        status.update(status='failed',error=repr(exc));raise
    finally:
        save()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--queue',action='store_true',help='8-update smoke passes before independent formal training')
    args=parser.parse_args()
    if args.queue:
        if args.smoke:
            parser.error('--queue and --smoke are mutually exclusive')
        run(smoke=True);run(smoke=False)
    else:
        run(smoke=args.smoke)
