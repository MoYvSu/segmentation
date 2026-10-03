# -*- coding: utf-8 -*-
"""人工原生窗口单变量：精确复用已封存50%旧坐标/50%九点坐标配方。"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

FORMAT = 'affinity_coverage_v1'
SEED, FLAG_NAMESPACE, ANCHOR_NAMESPACE = 20271002, 4103, 4109
METADATA_SHA = '33dd94a24ad720d9e52edaa83facf39733210593b9770b0d290a8978ec6c3b07'
TRIAL_SHA = '9a207cc9bd8953eb0d807c288c059a1aff951a06fd26c0a1507a163b4d56178b'
REFERENCE_SHA = '43da103c95c7f85149bd0533fd116d26f0a81c91d793db7e8938b7d4ae80e434'
REFERENCE_STATUS_SHA = '7feb9da496cc0a23ab944663cca4d071724b7008bb0757d84132b80edcaf5840'
REFERENCE_STEPS_SHA = 'b471f0896ce559db2e3517ded8cd872ecda0dbb3a9798b8f8b3ade0dcf83a6a3'
OPTIONS = dict(format=FORMAT,reference='outputs/affinity_balance/mixed',reference_sha256=REFERENCE_SHA,
    trial_root='outputs/affinity_crop_trial',trial_sha256=TRIAL_SHA,metadata_sha256=METADATA_SHA,
    seed=SEED,flag_namespace=FLAG_NAMESPACE,anchor_namespace=ANCHOR_NAMESPACE,
    per_epoch_old_uniform=16,per_epoch_anchor=16,manual_native_only=True,formal_requires_explicit_go=True)
FIXED_KEYS = ('epoch', 'step', 'seed', 'manual', 'pseudo', 'manual_draw', 'pseudo_draw',
              'manual_crop', 'pseudo_crop', 'crop_attempts', 'training_view', 'updates', 'auxiliary_weight')


def anchors(length, size):
    stop=max(0,int(length)-int(size))
    return sorted(set((0,round(stop/2),stop)))


def flags(epoch):
    if isinstance(epoch,bool) or not isinstance(epoch,(int,np.integer)) or not 1<=epoch<=20:
        raise ValueError('Actual training epoch 1..20 required')
    return np.random.default_rng(np.random.SeedSequence([SEED,int(epoch),FLAG_NAMESPACE])).permutation(
        np.repeat([False,True],16)).tolist()


def anchor_box(ids, epoch, update, *, size=1024, attempts=32, minimum_pairs=16):
    """种子里的update是实际更新号，不是source/draw索引；不消耗任何旧随机流。"""
    ids=np.asarray(ids)
    if ids.ndim!=2 or not ids.size or not np.issubdtype(ids.dtype,np.integer) or ids.min()<0:
        raise ValueError('Nonempty canonical integer instance map required')
    if min(size,attempts,minimum_pairs)<1 or isinstance(epoch,bool) or not 1<=epoch<=20:
        raise ValueError('Invalid crop recipe')
    if isinstance(update,bool) or not isinstance(update,(int,np.integer)) or not (epoch-1)*64<update<=epoch*64:
        raise ValueError('Actual update index must belong to this epoch')
    h,w=ids.shape;ch,cw=min(h,int(size)),min(w,int(size))
    ys,xs=anchors(h,ch),anchors(w,cw)
    rng=np.random.default_rng(np.random.SeedSequence([SEED,int(epoch),int(update),ANCHOR_NAMESPACE]))
    for attempt in range(int(attempts)):
        y,x=int(rng.choice(ys)),int(rng.choice(xs))
        known=ids[y:y+ch,x:x+cw]>0
        count=int((known[1:]&known[:-1]).sum()+(known[:,1:]&known[:,:-1]).sum())
        if count>=minimum_pairs:
            return [y,x,y+ch,x+cw],attempt,count
    raise RuntimeError('Selected source exhausted32anchor attempts; no fallback/no source replacement')


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file,'sha256').hexdigest()


class CoveragePlan:
    """从真实更新顺序生成路线；1280逐条比对旧控制及未训练反事实收据。"""
    def __init__(self, reference, candidate, *, epochs=20):
        if epochs!=20 or len(reference)!=1280 or len(candidate)!=1280:
            raise ValueError('Full original20epoch/1280draw plan required')
        self.reference,self.candidate=deepcopy(reference),deepcopy(candidate)
        self.by_draw={};self.by_update={}
        for epoch in range(1,21):
            rows=self.reference[(epoch-1)*64:epoch*64]
            if any(r['epoch']!=epoch or r['step']!=i+1 or r['updates']!=(epoch-1)*64+i+1 for i,r in enumerate(rows)):
                raise ValueError('Actual epoch/step/update ordering differs')
            native=[r for r in rows if r['training_view']=='native1024']
            if len(native)!=32 or sum(r['training_view']=='whole' for r in rows)!=32:
                raise ValueError('Exact32whole/32native schedule required')
            assignments={r['updates']:f for r,f in zip(native,flags(epoch))}
            for old in rows:
                new=self.candidate[old['updates']-1]
                if new.get('counterfactual_not_executed') is not True:
                    raise ValueError('Expected sealed unexecuted CPU candidate receipt')
                selected=assignments.get(old['updates'],False)
                for key in FIXED_KEYS:
                    if selected and key in ('manual_crop','crop_attempts'):
                        continue
                    if old[key]!=new[key]:
                        raise ValueError('Counterfactual changed a fixed field: '+key)
                tag=new.get('counterfactual_anchor')
                if old['training_view']=='native1024':
                    if not isinstance(tag,dict) or tag.get('selected') is not selected:
                        raise ValueError('Counterfactual route flag differs')
                    if not selected and tag != {'selected':False}:
                        raise ValueError('Uniform route must have only the false selection flag')
                    if selected and (set(tag)!= {'selected','accepted_attempt','known_cardinal_pairs'}
                            or new['crop_attempts'][0]!=[tag['accepted_attempt']]
                            or new['crop_attempts'][1]!=old['crop_attempts'][1]):
                        raise ValueError('Counterfactual accepted crop/pseudo attempt differs')
                elif tag is not None:
                    raise ValueError('Whole view cannot have an anchor route')
                index=int(old['manual_draw']['draw_index'][0]);key=(epoch,index)
                if key in self.by_draw:
                    raise ValueError('Duplicated actual manual draw')
                self.by_draw[key]=new;self.by_update[old['updates']]=new

    def row(self, epoch, draw):
        return self.by_draw[(int(epoch),int(draw))]

    def check_crop(self, ids, row, *, size, options):
        tag=row.get('counterfactual_anchor')
        if tag and tag['selected']:
            box,attempt,known=anchor_box(ids,row['epoch'],row['updates'],size=size,
                attempts=options['crop_attempts'],minimum_pairs=options['minimum_known_pairs'])
            if ([box]!=row['manual_crop'] or attempt!=tag['accepted_attempt'] or known!=tag['known_cardinal_pairs']):
                raise RuntimeError('Actual anchor selection differs from sealed1280counterfactual')
            return box,attempt,known
        return row['manual_crop'][0],row['crop_attempts'][0][0],None


def read_plan(config):
    from utils.config import project_path
    opt=config['affinity_coverage'];reference=Path(project_path(config,opt['reference']))/'steps.jsonl'
    if opt!=OPTIONS:
        raise ValueError('Only the fixed approved coverage recipe is supported')
    prior=json.loads((reference.parent/'status.json').read_text(encoding='utf8'))
    if (sha(reference.parent/'status.json')!=REFERENCE_STATUS_SHA or sha(reference)!=REFERENCE_STEPS_SHA
            or sha(reference.parent/'final.pt')!=REFERENCE_SHA or prior.get('status')!='complete'
            or prior.get('epoch')!=20 or prior.get('updates')!=1280):
        raise RuntimeError('Original R1 e20 status/1280 receipts/checkpoint differ')
    trial=Path(project_path(config,opt['trial_root']))
    if sha(trial/'report.json')!=TRIAL_SHA or sha(trial/'metadata_receipts.jsonl')!=METADATA_SHA:
        raise RuntimeError('Sealed CPU counterfactual metadata/recipe differs')
    report=json.loads((trial/'report.json').read_text(encoding='utf-8'))
    if not report['complete'] or report['seed']!=SEED or report['native640_anchor320_uniform320'] is not True:
        raise RuntimeError('Completed fixedcounterfactual required')
    old=[json.loads(s) for s in reference.read_text(encoding='utf-8').splitlines()]
    new=[json.loads(s) for s in (trial/'metadata_receipts.jsonl').read_text(encoding='utf-8').splitlines()]
    return CoveragePlan(old,new)


class CoverageNativeViewDataset(Dataset):
    """旧uniform直接委托原对象；新窗口沿用同源RGB/新GT/letterbox读取。"""
    def __init__(self, original, plan):
        if original.kind!='manual':
            raise ValueError('Only manual-native coordinates may change')
        self.original,self.plan=original,plan
        for name in ('base','kind','size','seed','options','image_size','grid'):
            setattr(self,name,getattr(original,name))
        self.epoch,self.draw=0,0

    def __len__(self):
        return len(self.original)

    def __getitem__(self, source_index):
        from data.affinity_native import geometry_sample
        from data.rgb_restoration_dataset import read_rgb
        from data.semantic_targets import load_completed_semantic_source
        row=self.plan.row(self.epoch,self.draw)
        if row['training_view']!='native1024' or row['manual_draw']['source_index']!=[int(source_index)]:
            raise RuntimeError('Actual native source/draw mapping differs from sealed schedule')
        tag=row['counterfactual_anchor'];self.original.epoch,self.original.draw=self.epoch,self.draw
        if not tag['selected']:
            sample=self.original[source_index]
        else:
            path=self.base.samples[source_index];image=read_rgb(str(path))
            ids,_=load_completed_semantic_source(self.base.completed_gt_dir/(path.stem+'_gt.npz'),image.shape[:2])
            if image.shape[:2]!=ids.shape:
                raise ValueError('Native canonical image/newGT differ')
            box,attempt,_=self.plan.check_crop(ids,row,size=self.size,options=self.options)
            y,x,y1,x1=box
            sample=geometry_sample(image[y:y1,x:x1],ids[y:y1,x:x1],image_size=self.image_size,grid=self.grid)
            sample.update(name=path.name,image_name=path.name,image_path=str(path),source_kind=self.kind,
                crop_box=torch.tensor(box,dtype=torch.int32),native_source_shape=torch.tensor(ids.shape,dtype=torch.int32),
                crop_attempt=attempt,native_crop_size=self.size)
        if sample['crop_box'].tolist()!=row['manual_crop'][0] or sample['crop_attempt']!=row['crop_attempts'][0][0]:
            raise RuntimeError('Real routed crop differs from sealed CPU candidate')
        sample.update(coverage_anchor=bool(tag['selected']),coverage_update=int(row['updates']))
        return sample


def build_datasets(config, size=1024, *, coverage=True):
    from data.affinity_mixed import build_datasets as original_builder
    if size!=1024 or coverage is not True:
        raise ValueError('Formal recipe is manual coverage at native1024')
    plan=read_plan(config)
    manual,pseudo,data=original_builder(config,size)
    manual.native.base_dataset=CoverageNativeViewDataset(manual.native.base_dataset,plan)
    result=deepcopy(data)
    result['coverage']=dict(format=FORMAT,seed=SEED,flag_namespace=FLAG_NAMESPACE,anchor_namespace=ANCHOR_NAMESPACE,
        manual_native_only=True,per_epoch_old_uniform=16,per_epoch_anchor=16,trial_sha256=TRIAL_SHA,
        metadata_sha256=METADATA_SHA,whole_and_pseudo_unchanged=True)
    return manual,pseudo,result
