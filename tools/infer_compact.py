# -*- coding: utf-8 -*-
"""原C/F0的输出等价后处理精简；默认C，前向、raw票和final200保持原路径。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from tools import infer_zero as zero
from tools import infer_flat_semantic as flat
from utils import compact_instance_readout as compact

entry,naf=zero.entry,zero.naf_entry
FORMAT='compact_instance_inference_v1'
AUDIT_FILES={'report.json','status.json','naf_report.json','naf_failed.json',
             'zero_report.json','zero_failed.json','flat_semantic_report.json','flat_semantic_failed.json'}


class PlainPatch:
    """旧wrapper的side_effect改为普通函数，不积存逐图数组调用历史。"""
    @staticmethod
    def object(target,name,*,side_effect=None,new=None):
        if side_effect is not None and new is not None:
            raise ValueError('Supply one replacement function')
        return patch.object(target,name,new=side_effect if side_effect is not None else new)


def contract(mode):
    return dict(format=compact.FORMAT,geometry_mode=mode,default_geometry='C',
        F0_is_separate_candidate=True,model_forward_unchanged=True,
        original_backend_sha256=flat.BACKEND_SHA,NAF_model_sha256=naf.EXPECTED_MODEL_SHA,
        marker_stage_calls_per_source=1,reflect32_marker_calls_per_source=1,S0_filter_calls_per_source=1,
        actual_WS_calls_per_source=1,temporary_semantic_vote_calls=0,
        temporary_classes='P0 placeholders, overwritten by unchanged raw probability_mean vote',
        raw_probability_and_float32_reduction_order_unchanged=True,
        original_pre_vote_area_cap_sort_and_uint16_mapping=True,final200_calls_per_source=1,
        no_noise_generator=True,training=False)


def audit_document(document,rows,state,source_sha,utility_sha,mode):
    value=deepcopy(document)
    value.setdefault('base_inference_format',value.get('format'))
    value.update(format=FORMAT,training=False,compact_readout=contract(mode),
        compact_entry_sha256=source_sha,compact_utility_sha256=utility_sha,
        compact_readout_images=deepcopy(rows),geometry_mode=mode)
    images=value.get('images',[])
    for index,row in enumerate(value['compact_readout_images']):
        if index<len(images):row['source']=images[index]['source']
    value['actual_compact_counts']=dict(marker_stage_calls=len(rows),reflect_marker_calls=len(rows),
        S0_filter_calls=len(rows),actual_WS_calls=len(rows),temporary_semantic_vote_calls=0,
        raw_revote_calls=state['raw_calls'],final200_calls=state['final_calls'])
    if 'config' in value:
        value['config']['compact_readout']=contract(mode)
    return value


def run(args,*,device='cuda'):
    args=argparse.Namespace(**vars(args))
    mode=getattr(args,'geometry_mode','C')
    if mode not in ('C','F0') or getattr(args,'semantic_checkpoint',None):
        raise ValueError('Choose original C or F0 geometry; trained semantic deltas are forbidden')
    # 两分支均锁原13b3/9315、N1/旧20a59/S0和seed50/final200。
    baseline=entry.load_config(str(Path(args.config).resolve()))
    if getattr(args,'dataset',None):baseline['submission']['dataset']=args.dataset
    flat.execution_config(baseline)
    output=Path(args.output_dir).resolve()
    if output.exists():raise FileExistsError('Use a fresh compact inference output')
    source_sha,utility_sha=entry.sha(__file__),entry.sha(compact.__file__)
    original_write,original_vote,original_final=entry.write,entry.views.revote_instances,entry.finalize_instance_area
    rows=[];state=dict(raw_calls=0,final_calls=0)

    def readout(semantic,after,reference,config):
        result,audit=compact.compact_supported_readout(semantic,after,reference,config,geometry_mode=mode)
        rows.append(dict(index=len(rows),shape=list(after.shape),
            semantic_sha256=compact.array_sha(semantic.detach().cpu().contiguous().numpy()),
            input_boundary_sha256=compact.array_sha(reference),output_boundary_sha256=compact.array_sha(after),
            readout_hashes=deepcopy(audit['readout_hashes']),compact_readout=deepcopy(audit['compact_readout']),
            S0_output_marker_sha256=audit['output_marker_sha256'],
            pre200_ID_sha256=compact.array_sha(result[0])))
        return result,audit

    def flat_readout(semantic,after,reference,config,*,original_readout=None):
        # 绕开F0诊断wrapper对marker_stages的额外一遍重算；proof由同一次构建给出。
        return readout(semantic,after,reference,config)

    def revote(*values,**options):
        state['raw_calls']+=1
        return original_vote(*values,**options)

    def finalize(*values,**options):
        state['final_calls']+=1
        return original_final(*values,**options)

    def write(path,document):
        path=Path(path)
        if path.parent==output/'audit' and path.name in AUDIT_FILES:
            document=audit_document(document,rows,state,source_sha,utility_sha,mode)
        return original_write(path,document)

    try:
        with patch.object(naf,'_supported_readout',new=readout), \
                patch.object(naf,'patch',new=PlainPatch),patch.object(zero,'patch',new=PlainPatch), \
                patch.object(flat,'flat_supported_readout',new=flat_readout), \
                patch.object(entry.views,'revote_instances',new=revote), \
                patch.object(entry,'finalize_instance_area',new=finalize),patch.object(entry,'write',new=write):
            result=(zero if mode=='C' else flat).run(args,device=device)
        images=result['images']
        if len(rows)!=len(images) or state['raw_calls']!=len(images) or state['final_calls']!=len(images):
            raise RuntimeError('Compact WS/raw vote/final200 must each execute once per source')
        for row,image in zip(rows,images):
            for key in ('semantic_sha256','input_boundary_sha256','output_boundary_sha256'):
                if row[key]!=image[key]:raise RuntimeError('Compact and NAF map provenance differ')
            if row['readout_hashes']!=image['seed_support']['readout_hashes']:
                raise RuntimeError('Compact and actual NAF WS hashes differ')
            row['source']=image['source']
        if entry.sha(__file__)!=source_sha or entry.sha(compact.__file__)!=utility_sha:
            raise RuntimeError('Compact source changed during inference')
        result=audit_document(result,rows,state,source_sha,utility_sha,mode)
        original_write(output/'audit/compact_report.json',result)
        return result
    except Exception as exc:
        if (output/'audit').is_dir():
            original_write(output/'audit/compact_failed.json',dict(format=FORMAT,complete=False,
                geometry_mode=mode,error=repr(exc),compact_readout_images=rows))
            for name in AUDIT_FILES:
                path=output/'audit'/name
                if path.is_file():
                    document=json.loads(path.read_text(encoding='utf-8'))
                    document.update(complete=False,compact_error=repr(exc))
                    if name=='status.json':document['status']='failed'
                    original_write(path,document)
        raise


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir',required=True);parser.add_argument('--output-dir',required=True)
    parser.add_argument('--config',default=str(ROOT/'config/inference/area_floor_200.yaml'))
    parser.add_argument('--model-dir');parser.add_argument('--dataset',choices=('quarter_final_100','semifinal_150'))
    parser.add_argument('--only',nargs='+');parser.add_argument('--naf-checkpoint',required=True)
    parser.add_argument('--capture-names',nargs='+');parser.add_argument('--global-noise',choices=('zero',),default='zero')
    parser.add_argument('--geometry-mode',choices=('C','F0'),default='C')
    return parser.parse_args(argv)


if __name__=='__main__':
    result=run(parse_args())
    print(json.dumps(dict(complete=True,format=FORMAT,geometry_mode=result['geometry_mode'],
        images=len(result['images']),actual_compact_counts=result['actual_compact_counts'])),flush=True)
