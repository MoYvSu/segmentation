# -*- coding: utf-8 -*-
"""原C整数读出加速候选；只替换finish_watershed，旧入口默认保持不变。"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import infer_compact as compact
from utils import fast_instance_readout as fast

entry = compact.entry
FORMAT = 'fast_integer_readout_inference_v1'
parse_args = compact.parse_args


def audit_document(document, state, source_sha, utility_sha, mode):
    """明确区分本候选与原compact执行，保留原读出的其余审计字段。"""
    value = deepcopy(document)
    value.setdefault('fast_readout_base_format', value.get('format'))
    contract = dict(
        format=fast.FORMAT, enabled=True, geometry_mode=mode, default_geometry='C',
        replacement='utils.fast_instance_readout.finish_watershed',
        reference='utils.compact_instance_readout.finish_watershed',
        only_integer_finish_replaced=True, original_cap_input_and_function_unchanged=True,
        model_forward_unchanged=True, raw_float32_vote_unchanged=True,
        final200_unchanged=True, no_training=True,
        finish_watershed_calls=state['calls'], completed_finish_calls=state['completed'])
    value.update(format=FORMAT, training=False, no_training=True,
        fast_readout=contract, fast_readout_entry_sha256=source_sha,
        fast_readout_utility_sha256=utility_sha)
    if isinstance(value.get('config'), dict):
        value['config']['fast_readout'] = deepcopy(contract)
    return value


def run(args, *, device='cuda'):
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh fast readout inference output')
    mode = getattr(args, 'geometry_mode', 'C')
    source_sha, utility_sha = entry.sha(__file__), entry.sha(fast.__file__)
    original_write = entry.write
    state = dict(calls=0, completed=0)

    def finish(watershed, inference):
        # 普通函数只保留计数；不保存逐图数组或Mock调用历史。
        state['calls'] += 1
        result = fast.finish_watershed(watershed, inference)
        state['completed'] += 1
        return result

    def write(path, document):
        path = Path(path)
        if path.parent.resolve() == output / 'audit' and path.suffix == '.json':
            document = audit_document(document, state, source_sha, utility_sha, mode)
        return original_write(path, document)

    try:
        with patch.object(compact.compact, 'finish_watershed', new=finish), \
                patch.object(entry, 'write', new=write):
            result = compact.run(args, device=device)
        if state['calls'] != len(result['images']) or state['completed'] != state['calls']:
            raise RuntimeError('Fast integer finish must execute exactly once per source')
        if entry.sha(__file__) != source_sha or entry.sha(fast.__file__) != utility_sha:
            raise RuntimeError('Fast readout source changed during inference')
        result = audit_document(result, state, source_sha, utility_sha, mode)
        original_write(output / 'audit/fast_readout_report.json', result)
        return result
    except Exception as exc:
        # 只处理本次新建输出，避免把旧成功报告留成此次候选的成功回执。
        audit_dir = output / 'audit'
        if audit_dir.is_dir():
            for path in sorted(audit_dir.glob('*.json')):
                document = json.loads(path.read_text(encoding='utf-8'))
                document.update(complete=False, fast_readout_error=repr(exc))
                if path.name == 'status.json':
                    document['status'] = 'failed'
                original_write(path, audit_document(
                    document, state, source_sha, utility_sha, mode))
            failed = dict(complete=False, geometry_mode=mode, error=repr(exc))
            original_write(audit_dir / 'fast_readout_failed.json', audit_document(
                failed, state, source_sha, utility_sha, mode))
        raise


def main(argv=None):
    result = run(parse_args(argv))
    print(json.dumps(dict(complete=bool(result.get('complete')), format=FORMAT,
        images=len(result['images']), fast_readout=result['fast_readout']),
        ensure_ascii=False), flush=True)
    return result


if __name__ == '__main__':
    main()
