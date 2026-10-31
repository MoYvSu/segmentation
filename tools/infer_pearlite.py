# -*- coding: utf-8 -*-
"""固定旧C配方，完成raw投票/final200后一次性合并相邻P，不提供实验开关。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import infer_boundary_rgb as naf_entry
from utils.pearlite_merge import merge_adjacent_pearlite

FORMAT = 'fixed_adjacent_pearlite_inference_v2'
NAF_CHECKPOINT_FORMAT = 'boundary_rgb_training_v1'
FIXED_CONFIG = ROOT/'config/inference/area_floor_200.yaml'
NAF_MODEL_SHA = '20a59c3971b2112c48a3b744c31f8051f5687772456cd35b5d38f0ae41b158e8'


def apply_result(rgb, result):
    """只改最终P实例及相应类别表；保留F、边界和存活ID原分数。"""
    ids, classes, audit = merge_adjacent_pearlite(result['instances'], result['classes'])
    naf_entry.entry.verify_prediction(ids, classes, rgb.shape[:2])
    output = dict(result, instances=ids, classes=classes)
    output['phases'] = naf_entry.entry.phase_description(ids, classes)
    if 'instance_scores' in result:
        output['instance_scores'] = {key: result['instance_scores'][key] for key in classes}
        output['instance_score_policy'] = 'pearlite_survivor_original_score_inherited_not_recomputed'
    return output, audit


def run(args, *, device='cuda'):
    """独立固定路线；原NAF入口与模型保持原义，不修改共享默认部署。"""
    if naf_entry.CHECKPOINT_FORMAT != NAF_CHECKPOINT_FORMAT:
        raise RuntimeError('固定20a59必须使用boundary_rgb_training_v1的严格入口')
    for key in ('relation_checkpoint', 'relation_threshold', 'relation_batch_size', 'relation_expected_model_sha'):
        if getattr(args, key, None) is not None:
            raise ValueError('固定P合并脚本不接受关系头或关系超参数')
    if getattr(args, 'config', None) is not None and Path(args.config).resolve() != FIXED_CONFIG.resolve():
        raise ValueError('固定P合并脚本只接受原C的area_floor_200配方')
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError('固定P候选须使用全新输出目录：'+str(output))
    config = naf_entry.entry.load_config(str(FIXED_CONFIG))
    area = naf_entry.entry.resolve_instance_area_policy(config['inference'])
    if (area['final_min_instance_area'] != 200 or area['final_small_instance_action'] != 'neighbor_merge_v1'
            or config.get('boundary_relation', {}).get('enabled', False)):
        raise ValueError('固定C终态或禁止关系头的配方发生变化')
    actual = argparse.Namespace(input_dir=args.input_dir, output_dir=args.output_dir,
        model_dir=getattr(args, 'model_dir', None), naf_checkpoint=args.naf_checkpoint,
        dataset=getattr(args, 'dataset', None) or 'semifinal_150', only=getattr(args, 'only', None),
        config=str(FIXED_CONFIG), seed_support='original-zero', save_analysis=False,
        capture_names=None, capture_all_maps=False, expected_model_sha=NAF_MODEL_SHA)
    source_files = ('tools/infer_pearlite.py', 'utils/pearlite_merge.py',
                    'tools/infer_boundary_rgb.py', 'config/inference/area_floor_200.yaml')
    source_sha = {name: naf_entry.entry.sha(ROOT/name) for name in source_files}
    original_load = naf_entry.load_naf_checkpoint
    original_predict = naf_entry.entry.predict
    records = []

    def load(*values, **options):
        if options.get('expected_model_sha') != NAF_MODEL_SHA:
            raise RuntimeError('固定P合并必须向真实RGB格式loader传递20a59 SHA')
        model, provenance = original_load(*values, **options)
        if (provenance['model_sha256'] != NAF_MODEL_SHA
                or provenance['checkpoint_format'] != NAF_CHECKPOINT_FORMAT):
            raise ValueError('固定P合并必须使用原20a59 NAF')
        return model, provenance

    def predict(model, restorer, terminal, path, config, actual_device, **options):
        rgb, result = original_predict(model, restorer, terminal, path, config, actual_device, **options)
        # original_predict在此已经完成raw类别投票与final200；只执行一次。
        started = time.perf_counter()
        final, audit = apply_result(rgb, result)
        elapsed = time.perf_counter()-started
        name = Path(path).stem
        if any(record['source'] == name for record in records):
            raise RuntimeError('固定P脚本不允许重复处理同一源')
        folder = output/'audit/pearlite'
        folder.mkdir(parents=True, exist_ok=True)
        audit_path = folder/(name+'.json')
        if audit_path.exists():
            raise FileExistsError(audit_path)
        naf_entry.entry.write(audit_path, audit)
        records.append(dict(source=name, seconds=elapsed,
            audit=audit_path.relative_to(output).as_posix(), audit_sha256=naf_entry.entry.sha(audit_path),
            removed_P_instance_count=audit['removed_P_instance_count'],
            filled_seam_pixels=audit['filled_seam_pixels'],
            F_final_ID_map_and_classes_exact=audit['F_final_ID_map_and_classes_exact']))
        return rgb, final

    try:
        with patch.object(naf_entry, 'load_naf_checkpoint', side_effect=load), \
                patch.object(naf_entry.entry, 'predict', side_effect=predict):
            upstream = naf_entry.run(actual, device=device)
        if (not upstream.get('complete') or [r['source'] for r in records] != [r['source'] for r in upstream['images']]
                or any(naf_entry.entry.sha(ROOT/name) != digest for name, digest in source_sha.items())):
            raise RuntimeError('固定P脚本源代码或完整图像清单发生变化')
        receipt = dict(complete=True, format=FORMAT, training=False, parameter_count=upstream['parameter_count'],
            added_parameter_count=0, naf_model_sha256=NAF_MODEL_SHA, fixed_config='config/inference/area_floor_200.yaml',
            upstream_interface='tools/infer_boundary_rgb.py', naf_checkpoint_format=NAF_CHECKPOINT_FORMAT,
            naf_input_contract=deepcopy(upstream['naf']['input_contract']),
            naf_numerical_contract=deepcopy(upstream['naf']['numerical_contract']),
            seed_support='original-zero', stage='after_raw_vote_and_final200', images=records,
            exactly_one_merge_per_source=True, no_relation_model=True, no_revote=True, no_second_area_cleanup=True,
            no_WS_rerun=True, F_final_ID_map_and_classes_exact=True,
            source_sha256=source_sha, upstream_naf_receipt='audit/naf_report.json',
            timing=dict(merge=naf_entry._timing_summary(records, 'seconds'),
                scope='CPU merge and result contract; excludes audit/result saves; already included in upstream predict timing'))
        audit = output/'audit'
        naf_entry.entry.write(audit/'pearlite_report.json', receipt)
        report_path = audit/'report.json'
        report = json.loads(report_path.read_text(encoding='utf8'))
        report['geometry'] += ' -> one original-adjacency P connected-group merge'
        report.update(pearlite_merge_complete=True, fixed_pearlite_recipe=deepcopy(receipt))
        naf_entry.entry.write(report_path, report)
        status_path = audit/'status.json'
        status = json.loads(status_path.read_text(encoding='utf8'))
        status['pearlite_merge_complete'] = True
        naf_entry.entry.write(status_path, status)
        return receipt
    except Exception as error:
        audit = output/'audit'
        if audit.is_dir():
            naf_entry.entry.write(audit/'pearlite_failed.json', dict(complete=False, format=FORMAT, error=repr(error)))
            for name in ('status.json', 'report.json'):
                path = audit/name
                if path.is_file():
                    value = json.loads(path.read_text(encoding='utf8'))
                    value.update(complete=False, pearlite_merge_complete=False, pearlite_merge_error=repr(error))
                    if name == 'status.json':
                        value['status'] = 'failed'
                    naf_entry.entry.write(path, value)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--naf-checkpoint', required=True)
    parser.add_argument('--model-dir')
    parser.add_argument('--dataset', choices=('quarter_final_100', 'semifinal_150'), default='semifinal_150')
    parser.add_argument('--only', nargs='+')
    receipt = run(parser.parse_args(argv))
    print(json.dumps(dict(complete=True, format=FORMAT, images=len(receipt['images']),
        parameter_count=receipt['parameter_count'])), flush=True)


if __name__ == '__main__':
    main()
