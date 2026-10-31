# -*- coding: utf-8 -*-
"""同源互补视野affinity适配；不改变冻结后端、既有主随机流或部署格式。"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from copy import deepcopy
import gc
import json
import math
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from data.affinity_pair import build_datasets, validate_weight
from data.mim_dataset import list_images
from models.backend_adaptation import build_backend, load_backend, restore_first, save_backend
from tools.affinity_connectivity_views import _inference_state
from tools.affinity_native_views import predict_views, save_training_draw
from tools.render_backend_ablation import overlay, text_line, write_png
from train_affinity_balance import comparable, verify_reference as verify_balance_reference
from train_affinity_connectivity import append_json, configure_training, draw_receipt, frozen_digest, get_restorer, initial_parity
from train_affinity_native import compare_draw, read
from train_backend_adaptation import sha, tensor_digest, write_json
from train_direct_semantic_affinity import compute_affinity_loss, move_batch, set_seed
from utils.affinity_deployment import crop_letterbox_output, prepare_image
from utils.config import load_config, project_path


def reference_config(config):
    result = deepcopy(config)
    options = result.pop('affinity_pair')
    if set(options) != {'reference', 'reference_sha256', 'zero_reference', 'auxiliary_weight',
                        'auxiliary_seed', 'diagnostic_updates', 'monitor_width'}:
        raise ValueError('Unknown complementary-view configuration keys')
    if (options['auxiliary_weight'] != .25 or options['diagnostic_updates'] != 8
            or result['backend_adaptation']['epochs'] != 20
            or options['auxiliary_seed'] == result['backend_adaptation']['seed']):
        raise ValueError('The authorized experiment is independent auxiliary 0.25, 20 epochs, first 8 diagnostics')
    result['backend_adaptation']['output_dir'] = 'outputs/affinity_balance/mixed'
    return result


def verify_reference(config, state=None):
    """先复核原balance全部依赖，再严格核验直接mixed balance控制。"""
    base = reference_config(config)
    legacy_state = None if state is None else {**state, 'data': state['data']['parent_data']}
    verify_balance_reference(base, legacy_state)
    opt = config['affinity_pair']
    folder = Path(project_path(config, opt['reference']))
    prior = read(folder/'status.json')
    if comparable(base) != comparable(prior['config']):
        raise RuntimeError('A shared configuration changed beyond the explicit complementary branch')
    if prior['status'] != 'complete' or prior['updates'] != 1280 or sha(folder/'final.pt') != opt['reference_sha256']:
        raise RuntimeError('Mixed balance control is incomplete or changed')
    for key in ('frozen_unchanged', 'restorer_unchanged', 'affinity_changed', 'strict_reload_equal'):
        if prior.get(key) is not True:
            raise RuntimeError('Mixed balance audit failed: ' + key)
    if state is not None:
        for key in ('initial_state_sha256', 'frozen_state_sha256', 'restorer_state_sha256'):
            if prior[key] != state[key]:
                raise RuntimeError('Shared initial/frozen dependency changed: ' + key)
        if prior['data'] != state['data']['parent_data']:
            raise RuntimeError('Primary sampling data differs from mixed balance')
    rows = [json.loads(line) for line in (folder/'steps.jsonl').read_text(encoding='utf8').splitlines()]
    if len(rows) != 1280 or any(row['auxiliary_weight'] != 0 for row in rows):
        raise RuntimeError('Invalid mixed balance reference receipts')
    return prior, rows


def capture_rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch_cpu=torch.get_rng_state(),
                torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    if state['torch_cuda']:
        if not torch.cuda.is_available() or len(state['torch_cuda']) != torch.cuda.device_count():
            raise RuntimeError('CUDA RNG device count differs from saved continuation state')
        torch.cuda.set_rng_state_all(state['torch_cuda'])


@contextmanager
def isolated_random_stream(seed):
    state = capture_rng_state()
    try:
        set_seed(int(seed))
        yield
    finally:
        restore_rng_state(state)


def branch_coefficients(alpha):
    alpha = validate_weight(alpha)
    return (1., 0.) if alpha == 0 else (1/(1+alpha), alpha/(1+alpha))


def pair_receipt(raw, raw_pseudo, epoch, step, seed):
    if raw['training_view'] != raw_pseudo['training_view']:
        raise RuntimeError('Manual and SAM2 views differ within a branch')
    return dict(epoch=epoch, step=step, seed=seed, manual=raw['name'], pseudo=raw_pseudo['name'],
                manual_image_path=raw['image_path'], pseudo_image_path=raw_pseudo['image_path'],
                manual_draw=draw_receipt(raw), pseudo_draw=draw_receipt(raw_pseudo),
                manual_crop=raw['crop_box'].tolist(), pseudo_crop=raw_pseudo['crop_box'].tolist(),
                crop_attempts=[raw['crop_attempt'].tolist(), raw_pseudo['crop_attempt'].tolist()],
                training_view=raw['training_view'][0])


def compare_primary(row, reference):
    compare_draw(row, reference)
    for key in ('training_view', 'manual_crop', 'pseudo_crop', 'crop_attempts'):
        if row[key] != reference[key]:
            raise RuntimeError('Original mixed primary branch differs: ' + key)


def compare_pair(primary, auxiliary):
    if {primary['training_view'], auxiliary['training_view']} != {'whole', 'native1024'}:
        raise RuntimeError('Auxiliary view must complement the primary view')
    for kind in ('manual', 'pseudo'):
        if primary[kind] != auxiliary[kind]:
            raise RuntimeError('Auxiliary changed source: ' + kind)
        if primary[kind+'_image_path'] != auxiliary[kind+'_image_path']:
            raise RuntimeError('Auxiliary changed source path: ' + kind)
        for key in ('source_index', 'draw_index'):
            if primary[kind+'_draw'][key] != auxiliary[kind+'_draw'][key]:
                raise RuntimeError('Auxiliary changed source/draw identity: ' + kind + '/' + key)


def gradient_diagnostics(parameters, primary, main_coefficient, auxiliary_coefficient):
    """记录两支实际累积参数梯度；不增加投影、裁剪或另一次更新。"""
    main_sq = aux_sq = total_sq = dot = torch.zeros((), device=parameters[0].device)
    for parameter, before in zip(parameters, primary):
        total = torch.zeros_like(parameter) if parameter.grad is None else parameter.grad.detach()
        main = before.float()
        auxiliary = total.float() - main
        main_sq = main_sq + main.square().sum()
        aux_sq = aux_sq + auxiliary.square().sum()
        total_sq = total_sq + total.float().square().sum()
        dot = dot + (main*auxiliary).sum()
    main_norm, aux_norm, total_norm = main_sq.sqrt(), aux_sq.sqrt(), total_sq.sqrt()
    values = dict(main_weighted_parameter_norm=float(main_norm), auxiliary_weighted_parameter_norm=float(aux_norm),
                  combined_parameter_norm=float(total_norm),
                  main_unweighted_parameter_norm=float(main_norm/main_coefficient),
                  auxiliary_unweighted_parameter_norm=float(aux_norm/auxiliary_coefficient) if auxiliary_coefficient else 0.,
                  auxiliary_to_main_weighted_norm=float(aux_norm/main_norm.clamp_min(1e-12)),
                  parameter_gradient_cosine=float(dot/(main_norm*aux_norm).clamp_min(1e-12)))
    if any(not math.isfinite(v) for v in values.values()):
        raise FloatingPointError('Nonfinite branch parameter gradient diagnostics')
    return values


def verify_diagnostics(rows, alpha):
    """技术门槛只拒绝不有限、无更新或零分支偷跑；余弦用来解释机制。"""
    values = []
    for row in rows[:8]:
        metrics = row['gradient_diagnostics']
        if (not metrics or any(not math.isfinite(v) for v in metrics.values())
                or metrics['main_weighted_parameter_norm'] <= 0
                or metrics['combined_parameter_norm'] <= 0
                or not -1.000001 <= metrics['parameter_gradient_cosine'] <= 1.000001
                or not math.isfinite(row['parameter_update_norm']) or row['parameter_update_norm'] <= 0
                or not math.isfinite(row['grad_norm']) or row['grad_norm'] <= 0
                or not row['head_state_sha256']):
            raise RuntimeError('Invalid first-eight parameter gradient/update diagnostics')
        if (alpha > 0 and metrics['auxiliary_weighted_parameter_norm'] <= 0
                or alpha == 0 and metrics['auxiliary_weighted_parameter_norm'] != 0):
            raise RuntimeError('Auxiliary parameter gradients differ from enabled branches')
        values.append(metrics['parameter_gradient_cosine'])
    if len(values) != 8:
        raise RuntimeError('Incomplete first-eight gradient diagnostics')
    return dict(passed=True, updates=8, parameter_cosine_mean=sum(values)/len(values),
                parameter_cosine_min=min(values), nonfinite_or_missing_updates=False,
                scope='mechanism and technical gate; not a performance ranking')


def verify_monitor(config, folder, epoch, image_count):
    folder = Path(folder)/'monitor'/f'epoch_{epoch:03d}'
    summary = read(folder/'summary.json')
    if len(summary['images']) != image_count or summary['epoch'] != epoch:
        raise RuntimeError('Incomplete process thumbnail metadata')
    width = int(config['affinity_pair']['monitor_width']) * 4
    for row in summary['images']:
        image = cv2.imread(str(folder/(row['source']+'.png')), cv2.IMREAD_UNCHANGED)
        if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or image.shape[1] != width:
            raise RuntimeError('Missing or invalid four-column process thumbnail')


def save_monitor(model, restorer, config, device, output, epoch, smoke=False):
    """固定原图/D5a全图指引/原生边界/完整最终实例四列；不参与选模。"""
    names = config['backend_adaptation']['monitor']['images'][:2] if smoke else config['backend_adaptation']['monitor']['images']
    files = list(map(Path, list_images(project_path(config, config['inference']['test_dir']))))
    if not set(names).issubset({path.stem for path in files}):
        raise RuntimeError('Fixed monitor sources are incomplete')
    root = Path(output)/'monitor'/f'epoch_{epoch:03d}'
    root.mkdir(parents=True, exist_ok=True)
    width = int(config['affinity_pair']['monitor_width'])
    rows = []
    with _inference_state(model, restorer, device):
        for index, path in enumerate(files):
            if path.stem not in names:
                continue
            seed = config['backend_adaptation']['restoration']['inference_seed']+index
            rgb, views = predict_views(model, restorer, path, config, device, seed, [1024])
            pred = views['patch1024']
            _, image, ph, pw = prepare_image(path, 1024, device)
            restored = restore_first(restorer, image.float(), seed)
            restored = crop_letterbox_output(restored, 1024, ph, pw, rgb.shape[:2]).cpu()[0].permute(1,2,0).numpy()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            size = (width, max(1, round(rgb.shape[0]*width/rgb.shape[1])))
            image_small = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
            restored_small = cv2.resize(cv2.cvtColor(np.rint(np.clip(restored,0,1)*255).astype(np.uint8),cv2.COLOR_RGB2BGR),size)
            edge = cv2.resize(pred['boundary'],size,interpolation=cv2.INTER_AREA)
            edge = cv2.applyColorMap(np.rint(np.clip(edge,0,1)*255).astype(np.uint8),cv2.COLORMAP_INFERNO)
            ids = cv2.resize(pred['instances'],size,interpolation=cv2.INTER_NEAREST)
            final = overlay(image_small,ids,{int(k):v for k,v in pred['classes'].items()})
            panels = []
            for title, body in [('Original',image_small),('D5a whole guide',restored_small),('Affinity native1024',edge),(f'Final e{epoch}',final)]:
                panel = cv2.copyMakeBorder(body,30,0,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
                text_line(panel,title,(4,20),width,size=.4)
                panels.append(panel)
            write_png(root/(path.stem+'.png'),np.concatenate(panels,axis=1))
            rows.append(dict(source=path.stem,seed=seed,instances=len(pred['classes']),blend=pred['blend']))
    if len(rows) != len(names):
        raise RuntimeError('Process monitor cohort is incomplete')
    write_json(root/'summary.json',dict(epoch=epoch,images=rows,scope='fixed unlabeled thumbnails, no validation or selection'))


def train(config, output, smoke=False, auxiliary_weight=None):
    if not torch.cuda.is_available():
        raise RuntimeError('Use the GPU server sam2_env')
    reference_config(config)
    alpha = validate_weight(config['affinity_pair']['auxiliary_weight'] if auxiliary_weight is None else auxiliary_weight)
    if alpha == 0 and not smoke:
        raise ValueError('Zero auxiliary is a bounded technical smoke, never a retrained full control')
    cfg, opt = config['backend_adaptation'], config['affinity_pair']
    if cfg['num_workers'] != 0:
        raise ValueError('Pair draw state currently requires num_workers=0')
    torch.set_num_threads(4);cv2.setNumThreads(2)
    output = Path(output);output.mkdir(parents=True,exist_ok=False)
    device = torch.device('cuda')
    set_seed(cfg['seed'])
    model, initialization = build_backend(config,device)
    restorer = get_restorer(config,device)
    total = sum(p.numel() for p in model.parameters())+sum(p.numel() for p in restorer.parameters())
    if total >= 500000000:
        raise RuntimeError('Combined model parameter limit exceeded')
    manual, pseudo, data = build_datasets(config,auxiliary_weight=alpha)
    status = dict(status='preflight',smoke=smoke,config=config,initialization=initialization,data=data,
                  initial_state_sha256=tensor_digest(model.state_dict().items()),frozen_state_sha256=frozen_digest(model),
                  restorer_state_sha256=tensor_digest(restorer.state_dict().items()),total_parameters=total,
                  auxiliary_weight_actual=alpha,precision='FP32',selection='fixed_e20_all_sources_no_holdout',
                  control_retrained=False,official_score=None)
    write_json(output/'status.json',status)
    _, reference = verify_reference(config,status)
    status['reused_control_verified'] = True
    status['initial_parity'] = initial_parity(model,restorer,config,device)
    configure_training(model)
    parameters = list(model.affinity_decoder.parameters())
    initial_head = tensor_digest(model.affinity_decoder.state_dict().items())
    optimizer = torch.optim.AdamW(parameters,lr=cfg['learning_rates']['affinity'],weight_decay=cfg['weight_decay'],eps=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lambda e:cfg['minimum_lr_ratio']+(1-cfg['minimum_lr_ratio'])*(1+math.cos(math.pi*e/cfg['epochs']))/2)
    save_monitor(model,restorer,config,device,output,0,smoke)
    status['status']='running';write_json(output/'status.json',status)
    main_coefficient,auxiliary_coefficient = branch_coefficients(alpha)
    loss_config=config['direct_semantic_affinity']['affinity_loss']
    updates=0;started=time.time();epochs=1 if smoke else cfg['epochs']
    for epoch in range(1,epochs+1):
        for ds in (manual,pseudo):ds.set_epoch(epoch)
        loaders=[DataLoader(ds,batch_size=1,shuffle=True,num_workers=0,pin_memory=True,
                    generator=torch.Generator().manual_seed(cfg['seed']+epoch*2+i)) for i,ds in enumerate((manual,pseudo))]
        configure_training(model);sums=np.zeros(4);count=0
        for index,(pair_manual,pair_pseudo) in enumerate(zip(*loaders)):
            if smoke and index >= config['affinity_native']['smoke_updates']:break
            seed=cfg['seed']+epoch*10000+index;set_seed(seed)
            primary=pair_receipt(pair_manual['main'],pair_pseudo['main'],epoch,index+1,seed)
            compare_primary(primary,reference[updates])
            auxiliary=None;aux_seed=opt['auxiliary_seed']+epoch*10000+index
            if alpha:
                auxiliary=pair_receipt(pair_manual['auxiliary'],pair_pseudo['auxiliary'],epoch,index+1,aux_seed)
                compare_pair(primary,auxiliary)
            optimizer.zero_grad(set_to_none=True)
            raw_losses=[];supervision=[]
            diagnostic=updates < opt['diagnostic_updates']
            main_gradients=None
            for branch,coefficient in [('main',main_coefficient)]+([('auxiliary',auxiliary_coefficient)] if alpha else []):
                context=isolated_random_stream(aux_seed) if branch=='auxiliary' else nullcontext()
                with context:
                    for is_pseudo,paired in [(False,pair_manual),(True,pair_pseudo)]:
                        batch=move_batch(paired[branch],device)
                        restoration_seed=(seed+(2000000 if is_pseudo else 1000000) if branch=='main'
                                          else aux_seed+(4000000 if is_pseudo else 3000000))
                        with torch.no_grad():
                            restored=restore_first(restorer,batch['image'],restoration_seed)
                            if not is_pseudo and index<2 and (smoke or epoch==1 or epoch%5==0):
                                save_training_draw(output/'paired_draw'/branch,batch,restored,epoch,index+1)
                            features=model.encoder(restored)
                        logits=model.affinity_decoder(features)['affinity_logits']
                        loss,details=compute_affinity_loss(logits,batch,loss_config,pseudo=is_pseudo)
                        if not torch.isfinite(loss) or details['positive_edges']+details['negative_edges']<=0:
                            raise FloatingPointError('Nonfinite or unsupervised branch loss')
                        scale=cfg['pseudo_weight'] if is_pseudo else 1.
                        # alpha0 uses precisely the original operation, including the manual *1 multiplication.
                        multiplier=scale if alpha==0 else scale*coefficient
                        (loss*multiplier).backward()
                        raw_losses.append(float(loss.detach()))
                        supervision.append({k:details[k] for k in ('positive_edges','negative_edges','precision','recall','specificity')})
                        del loss,logits,restored,features,batch
                if branch=='main' and diagnostic:
                    main_gradients=[torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in parameters]
            norms=gradient_diagnostics(parameters,main_gradients,main_coefficient,auxiliary_coefficient) if diagnostic else None
            if main_gradients is not None:del main_gradients
            grad=float(torch.nn.utils.clip_grad_norm_(parameters,cfg['grad_clip'],error_if_nonfinite=True))
            if grad<=0 or any(p.grad is not None for n,p in model.named_parameters() if not n.startswith('affinity_decoder.')):
                raise RuntimeError('Missing affinity gradient or unexpected frozen gradient')
            before=[p.detach().clone() for p in parameters] if diagnostic else None
            optimizer.step()
            update_norm=None
            if before is not None:
                update_norm=float(torch.stack([(p.detach().float()-v.float()).square().sum() for p,v in zip(parameters,before)]).sum().sqrt())
                del before
            updates+=1;count+=1
            if alpha==0:raw_losses += [0.,0.]
            sums+=raw_losses
            row={**primary,'updates':updates,'manual_bce':raw_losses[0],'pseudo_bce':raw_losses[1],
                 'supervision':supervision[:2],'auxiliary_weight':alpha,'auxiliary':auxiliary,
                 'auxiliary_manual_bce':raw_losses[2],'auxiliary_pseudo_bce':raw_losses[3],
                 'auxiliary_supervision':supervision[2:],'main_coefficient':main_coefficient,
                 'auxiliary_coefficient':auxiliary_coefficient,'grad_norm':grad,'gradient_diagnostics':norms,
                 'parameter_update_norm':update_norm,
                 'head_state_sha256':tensor_digest(model.affinity_decoder.state_dict().items()) if diagnostic else None}
            append_json(output/'steps.jsonl',row)
            if index==0 or smoke:print(json.dumps(row),flush=True)
        if not smoke and count!=data['updates_per_epoch']:raise RuntimeError('Incomplete pair epoch')
        scheduler.step()
        epoch_row=dict(epoch=epoch,updates=updates,manual_bce=sums[0]/count,pseudo_bce=sums[1]/count,
                       auxiliary_manual_bce=sums[2]/count,auxiliary_pseudo_bce=sums[3]/count,
                       next_lr=optimizer.param_groups[0]['lr'],elapsed_seconds=time.time()-started,
                       peak_cuda_mib=torch.cuda.max_memory_allocated()/1024**2)
        append_json(output/'epochs.jsonl',epoch_row)
        checkpoint=dict(format='affinity_pair_head_v1',epoch=epoch,updates=updates,
                        geometry_state_dict=model.affinity_decoder.state_dict(),initialization=initialization,
                        config=config,auxiliary_weight=alpha,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                        rng_state=capture_rng_state(),data=data)
        torch.save(checkpoint,output/'last_head.pt')
        if epoch%5==0:torch.save({k:v for k,v in checkpoint.items() if k not in ('optimizer','scheduler','rng_state')},output/f'head_{epoch:03d}.pt')
        if epoch==1 or epoch%5==0 or epoch==epochs:save_monitor(model,restorer,config,device,output,epoch,smoke)
        status.update(status='running',**epoch_row);write_json(output/'status.json',status)
    status.update(frozen_unchanged=frozen_digest(model)==status['frozen_state_sha256'],
                  restorer_unchanged=tensor_digest(restorer.state_dict().items())==status['restorer_state_sha256'],
                  affinity_changed=tensor_digest(model.affinity_decoder.state_dict().items())!=initial_head,
                  paired_control_updates=updates,final_head_sha256=tensor_digest(model.affinity_decoder.state_dict().items()))
    if not all(status[k] for k in ('frozen_unchanged','restorer_unchanged','affinity_changed')):
        raise RuntimeError('Frozen/trainable state audit failed')
    final_digest=tensor_digest(model.state_dict().items())
    save_backend(model,config,initialization,output/'final.pt',epochs,
                 extra=dict(updates=updates,smoke=smoke,auxiliary_weight=alpha,data=data,
                            deployment='frozen_whole_semantic_stitched_native_affinity',continuation='last_head.pt'))
    del checkpoint,model,optimizer;gc.collect();torch.cuda.empty_cache()
    reloaded,_=load_backend(output/'final.pt',config,device)
    if tensor_digest(reloaded.state_dict().items())!=final_digest:raise RuntimeError('Strict final reload differs')
    status.update(status='complete',strict_reload_equal=True,final_state_sha256=final_digest,
                  final_checkpoint_sha256=sha(output/'final.pt'))
    write_json(output/'status.json',status)
    print(f'COMPLETE affinity pair {updates} updates alpha={alpha}',flush=True)


def verify_run(config, folder, expected):
    folder=Path(folder);state=read(folder/'status.json')
    _,reference=verify_reference(config,state)
    if state['status']!='complete' or state['updates']!=expected or state['config']!=json.loads(json.dumps(config)):
        raise RuntimeError('Pair training incomplete or actual configuration changed')
    for key in ('reused_control_verified','frozen_unchanged','restorer_unchanged','affinity_changed','strict_reload_equal'):
        if state.get(key) is not True:raise RuntimeError('Pair audit failed: '+key)
    alpha=state['auxiliary_weight_actual'];validate_weight(alpha)
    rows=[json.loads(line) for line in (folder/'steps.jsonl').read_text(encoding='utf8').splitlines()]
    if len(rows)!=expected:raise RuntimeError('Missing pair receipts')
    totals={'whole':0,'native1024':0}
    for row,old in zip(rows,reference):
        compare_primary(row,old)
        if row['auxiliary_weight']!=alpha or tuple(row[k] for k in ('main_coefficient','auxiliary_coefficient'))!=branch_coefficients(alpha):
            raise RuntimeError('Branch loss weights changed')
        totals[row['training_view']]+=1
        if alpha:compare_pair(row,row['auxiliary'])
        elif row['auxiliary'] is not None or row['auxiliary_supervision']:
            raise RuntimeError('Zero replay unexpectedly executed the auxiliary branch')
    if expected==1280 and totals!={'whole':640,'native1024':640}:raise RuntimeError('Mixed primary view budget changed')
    diagnostics=verify_diagnostics(rows,alpha)
    if sha(folder/'final.pt')!=state['final_checkpoint_sha256']:raise RuntimeError('Pair final checkpoint changed')
    continuation=torch.load(folder/'last_head.pt',map_location='cpu',weights_only=False)
    if not {'optimizer','scheduler','rng_state','geometry_state_dict','epoch','updates'}.issubset(continuation):
        raise RuntimeError('Continuation state is incomplete')
    if continuation['updates']!=expected or tensor_digest(continuation['geometry_state_dict'].items())!=state['final_head_sha256']:
        raise RuntimeError('Continuation state differs from the final head')
    monitor_epochs=[0,1] if state['smoke'] else [0,1,5,10,15,20]
    image_count=2 if state['smoke'] else len(config['backend_adaptation']['monitor']['images'])
    for epoch in monitor_epochs:
        verify_monitor(config,folder,epoch,image_count)
    return dict(passed=True,updates=expected,view_updates=totals,auxiliary_weight=alpha,
                primary_stream_equal=True,same_source_complementary_views=alpha>0,control_retrained=False,
                continuation_complete=True,monitor_complete=True,gradient_diagnostics=diagnostics,
                initialization_sha256=state['initial_state_sha256'])


ZERO_NUMERIC_CAPS = dict(loss=1e-5, gradient=2e-4, tensor_max_abs=5e-6, tensor_rms=2e-7)


def zero_difference(first_rows, second_rows, first_head, second_head):
    """逐项精确核对输入/监督计数，再报告浮点轨迹和每个参数tensor的差异。"""
    if len(first_rows) != 8 or len(second_rows) != 8 or set(first_head) != set(second_head):
        raise RuntimeError('Zero technical receipts or head keys are incomplete')
    steps = []
    exact = True
    for first, second in zip(first_rows, second_rows):
        compare_primary(first, second)
        if first['auxiliary_weight'] != 0 or second['auxiliary_weight'] != 0:
            raise RuntimeError('A zero technical reference executed an auxiliary branch')
        if len(first['supervision']) != 2 or len(second['supervision']) != 2:
            raise RuntimeError('Missing manual/SAM2 supervision counts')
        metric_difference = 0.
        for a, b in zip(first['supervision'], second['supervision']):
            for key in ('positive_edges', 'negative_edges'):
                if a[key] != b[key]:
                    raise RuntimeError('Zero technical supervision counts differ: ' + key)
            for key in ('precision', 'recall', 'specificity'):
                if not math.isfinite(a[key]) or not math.isfinite(b[key]):
                    raise RuntimeError('Nonfinite zero technical supervision metric')
                metric_difference = max(metric_difference, abs(a[key]-b[key]))
        row = dict(step=first['step'], supervision_metrics_max_abs=metric_difference)
        for key in ('manual_bce', 'pseudo_bce', 'grad_norm'):
            if not math.isfinite(first[key]) or not math.isfinite(second[key]):
                raise RuntimeError('Nonfinite zero technical loss/gradient')
            row[key] = abs(first[key]-second[key])
        exact = exact and metric_difference == 0 and all(row[k] == 0 for k in ('manual_bce','pseudo_bce','grad_norm'))
        steps.append(row)
    tensors = {}
    for name in first_head:
        a, b = first_head[name].detach().cpu(), second_head[name].detach().cpu()
        if a.dtype != b.dtype or a.shape != b.shape:
            raise RuntimeError('Zero technical head shape/dtype differs: ' + name)
        equal = torch.equal(a, b)
        if not a.is_floating_point():
            if not equal:
                raise RuntimeError('Zero technical nonfloating buffer differs: ' + name)
            maximum = rms = 0.
        else:
            if a.dtype != torch.float32 or not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise RuntimeError('Zero technical head is not finite FP32: ' + name)
            delta = a.double()-b.double()
            maximum, rms = float(delta.abs().max()), float(delta.square().mean().sqrt())
        tensors[name] = dict(max_abs=maximum, rms=rms, exact=equal)
        exact = exact and equal
    return dict(exact=exact, primary_receipts_exact=True, supervision_counts_exact=True,
                loss_max_abs=max(max(r['manual_bce'],r['pseudo_bce']) for r in steps),
                gradient_max_abs=max(r['grad_norm'] for r in steps),
                head_max_abs=max(v['max_abs'] for v in tensors.values()),
                head_max_rms=max(v['rms'] for v in tensors.values()), steps=steps, tensors=tensors)


def repeatability_limits(observed):
    """只接受本来很小的旧路径复放波动；3倍是工程裕量，不是置信区间。"""
    caps = ZERO_NUMERIC_CAPS
    if (observed['loss_max_abs'] > caps['loss'] or observed['gradient_max_abs'] > caps['gradient']
            or observed['head_max_abs'] > caps['tensor_max_abs'] or observed['head_max_rms'] > caps['tensor_rms']):
        raise RuntimeError('Legacy repeatability itself exceeds fixed absolute caps')
    if observed['exact']:
        raise RuntimeError('An exact legacy replay does not justify a numerical fallback')
    return dict(loss=min(caps['loss'],3*observed['loss_max_abs']),
                gradient=min(caps['gradient'],3*observed['gradient_max_abs']),
                tensors={name:dict(max_abs=min(caps['tensor_max_abs'],max(3*v['max_abs'],torch.finfo(torch.float32).eps)),
                                  rms=min(caps['tensor_rms'],max(3*v['rms'],1e-8)))
                         for name,v in observed['tensors'].items()},
                fixed_absolute_caps=dict(caps), engineering_margin=3., statistical_confidence_interval=False)


def verify_zero_numeric(difference, limits):
    for row in difference['steps']:
        if max(row['manual_bce'],row['pseudo_bce']) > limits['loss'] or row['grad_norm'] > limits['gradient']:
            raise RuntimeError('Zero trajectory exceeds measured repeatability limits')
    if set(difference['tensors']) != set(limits['tensors']):
        raise RuntimeError('Numerical fallback head keys changed')
    for name, values in difference['tensors'].items():
        if values['max_abs'] > limits['tensors'][name]['max_abs'] or values['rms'] > limits['tensors'][name]['rms']:
            raise RuntimeError('Zero head exceeds measured repeatability limits: ' + name)


def zero_reference(config, folder, state):
    """仅读取已完成的旧8步产物；审计绝不自动启动训练。"""
    folder = Path(folder)
    prior = read(folder/'status.json')
    if (prior['status'] != 'complete' or prior['updates'] != 8 or not prior['smoke']
            or comparable(prior['config']) != comparable(reference_config(config))):
        raise RuntimeError('Old eight-step technical reference is incompatible')
    for key in ('initial_state_sha256','frozen_state_sha256','restorer_state_sha256'):
        if state[key] != prior[key]:
            raise RuntimeError('Zero reference shared state differs: '+key)
    for key in ('frozen_unchanged','restorer_unchanged','affinity_changed','strict_reload_equal'):
        if prior.get(key) is not True:
            raise RuntimeError('Zero reference state audit failed: '+key)
    rows = [json.loads(v) for v in (folder/'steps.jsonl').read_text(encoding='utf8').splitlines()]
    checkpoint = torch.load(folder/'last_head.pt',map_location='cpu',weights_only=False)
    if checkpoint['epoch'] != 1 or comparable(checkpoint['config']) != comparable(prior['config']):
        raise RuntimeError('Zero reference checkpoint metadata differs')
    return rows, checkpoint['geometry_state_dict'], dict(folder=str(folder),
        status_sha256=sha(folder/'status.json'),steps_sha256=sha(folder/'steps.jsonl'),head_sha256=sha(folder/'last_head.pt'))


def audit_zero(config, folder, expected=8, *, repeatability_reference='outputs/pair_setup/legacy8'):
    """先尝试历史逐值兼容；非确定CUDA仅凭已测旧路径小波动作有界数值审计。"""
    folder=Path(folder);state=read(folder/'status.json')
    if expected!=8 or state['auxiliary_weight_actual']!=0 or not state['smoke']:
        raise ValueError('Zero parity is exactly eight technical steps')
    verify_run(config,folder,expected)
    source=Path(project_path(config,config['affinity_pair']['zero_reference']))
    a=[json.loads(v) for v in (folder/'steps.jsonl').read_text(encoding='utf8').splitlines()]
    head=torch.load(folder/'last_head.pt',map_location='cpu',weights_only=False)['geometry_state_dict']
    b,history_head,history_receipt=zero_reference(config,source,state)
    history_pair=zero_difference(b,a,history_head,head)
    result=dict(passed=True,updates=expected,loss_gradient_and_final_head_exact=history_pair['exact'],
                reference=str(source),reference_receipts={'history':history_receipt},
                comparisons={'history_vs_pair':history_pair},
                technical_replayed_updates=0,control_full_retrained=False,
                scope='technical compatibility; no official performance claim')
    if history_pair['exact']:
        result['compatibility_mode']='historical_bitwise_exact'
    else:
        repeat=Path(project_path(config,repeatability_reference))
        if repeat.resolve() in {source.resolve(),folder.resolve()}:
            raise RuntimeError('Repeatability requires an independent old-path eight-step replay')
        c,repeat_head,repeat_receipt=zero_reference(config,repeat,state)
        history_repeat=zero_difference(b,c,history_head,repeat_head)
        pair_repeat=zero_difference(a,c,head,repeat_head)
        limits=repeatability_limits(history_repeat)
        verify_zero_numeric(history_pair,limits)
        verify_zero_numeric(pair_repeat,limits)
        result.update(compatibility_mode='bounded_observed_cuda_repeatability',numeric_limits=limits,
                      historical_bitwise_exact=False,
                      legacy_replayed_updates_for_repeatability=8,
                      explanation='3x observed engineering margin with fixed caps, not a confidence interval')
        result['comparisons'].update(history_vs_legacy=history_repeat,pair_vs_legacy=pair_repeat)
        result['reference_receipts']['legacy']=repeat_receipt
    write_json(folder/'zero_parity.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='config/train/affinity_pair.yaml')
    parser.add_argument('--output',required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--auxiliary-weight',type=float,choices=[0.,.25])
    parser.add_argument('--audit-zero',action='store_true')
    args=parser.parse_args();config=load_config(args.config)
    if args.audit_zero:print(json.dumps(audit_zero(config,args.output),ensure_ascii=False))
    else:train(config,args.output,smoke=args.smoke,auxiliary_weight=args.auxiliary_weight)
