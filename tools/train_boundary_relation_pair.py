# -*- coding: utf-8 -*-
"""前台顺序运行 maps/RGB 关系分类对照，只保存输入引用及配对审计。

本工具不创建初态、不恢复旧训练、不复制数据或权重，也不应用区域合并。
config/data/initial 的相对路径基于 --code-root；output 基于调用者当前目录。
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid


ROOT = Path(__file__).resolve().parents[1]
CODE_FILES = ('train_boundary_relation.py', 'models/boundary_relation.py',
              'data/boundary_relation.py', 'utils/config.py')
RESERVED = ('maps', 'rgb', 'maps.log', 'rgb.log', 'queue_status.json',
            'queue_reference.json', 'queue_contract.json', 'paired_contract.json')
STEP_FIELDS = ('global_step', 'epoch', 'batch_in_epoch', 'sample_ids', 'samples',
               'keep', 'merge', 'shared_draw_sha256', 'learning_rate')


def utc():
    return datetime.now(timezone.utc).isoformat()


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False)


def canonical_sha(value):
    return hashlib.sha256(canonical(value).encode('utf8')).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write_json(path, value):
    path = Path(path)
    pending = path.with_name(path.name + '.pending.' + uuid.uuid4().hex)
    try:
        with pending.open('x', encoding='utf8') as stream:
            stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        pending.replace(path)
    finally:
        if pending.exists():
            pending.unlink()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value, name):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None,
            name + ' 必须是小写SHA256')
    return value


def positive_int(value, name):
    require(type(value) is int and value > 0, name + ' 必须是正整数')
    return value


def resolve_input(value, code_root):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else code_root / path).resolve()


def config_chain(path):
    """先检查继承循环，随后仍由项目 load_config 解析正式配置。"""
    import yaml
    rows, active = [], set()

    def visit(current):
        current = current.resolve()
        require(current not in active, '配置 _base 存在循环')
        active.add(current)
        payload = yaml.safe_load(current.read_text(encoding='utf8')) or {}
        require(isinstance(payload, dict), '配置必须是对象')
        base = payload.get('_base')
        if base:
            require(isinstance(base, str), '_base 必须是一个文件路径')
            visit(resolve_input(base, current.parent))
        rows.append(dict(path=str(current), sha256=file_sha(current)))
        active.remove(current)

    visit(Path(path))
    return rows


def preflight(args):
    code = Path(args.code_root).expanduser().resolve()
    config = resolve_input(args.config, code)
    data = resolve_input(args.data_manifest, code)
    initial = resolve_input(args.initial_load, code)
    output = Path(args.output).expanduser().resolve()
    for name in RESERVED:
        require(not os.path.lexists(output / name), '拒绝覆盖已有正式产物：' + str(output / name))
    chain = config_chain(config)
    spec = importlib.util.spec_from_file_location('_relation_pair_config', code / 'utils/config.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    resolved = module.load_config(str(config))
    cfg = resolved['boundary_relation']
    target = positive_int(cfg.get('max_updates'), 'max_updates')
    schedule = positive_int(cfg.get('schedule_updates'), 'schedule_updates')
    require(schedule >= target, 'schedule_updates 不能短于 max_updates')
    batch = positive_int(cfg.get('batch_size'), 'batch_size')
    start_lr, end_lr = float(cfg['learning_rate']), float(cfg['minimum_learning_rate'])
    require(math.isfinite(start_lr) and math.isfinite(end_lr) and 0 < end_lr <= start_lr,
            'cosine学习率端点无效')
    require(cfg.get('apply_predicted_merges') is False, '队列只允许关系评分，不应用合并')
    for key in ('monitor_updates', 'checkpoint_updates'):
        values = cfg.get(key)
        require(isinstance(values, list) and values == sorted(set(values)) and
                all(type(v) is int and 0 <= v <= target for v in values) and
                0 in values and target in values, key + ' 必须包含0和终点且无重复')
    manifest = read_json(data)
    require(manifest.get('complete') is True and manifest.get('format') == 'boundary_relation_data_v1',
            '正式数据manifest不完整或格式不符')
    require(manifest.get('training_eligibility', {}).get('eligible_by_count_only') is True,
            '数据尚未通过约定的数量/来源门槛')
    labels = {}
    for shard in manifest['shards']:
        require(shard['count'] == len(shard['samples']), '分片样本计数不符')
        for row in shard['samples']:
            require(row['label'] in (0, 1) and row['sample_id'] not in labels,
                    '训练标签必须为0/1且sample ID唯一')
            labels[row['sample_id']] = row['label']
    require(set(labels.values()) == {0, 1}, '两类可靠训练标签均不能为空')
    monitor_labels = {sid: labels[sid] for sid in manifest.get('monitor', {}).get('train_sample_ids', [])}
    for shard in manifest.get('monitor_shards', []):
        for row in shard['samples']:
            require(row['label'] == 255 and row['sample_id'] not in monitor_labels,
                    '测试monitor必须独立且仅含255标签')
            monitor_labels[row['sample_id']] = None
    code_files = {relative: file_sha(code / relative) for relative in CODE_FILES}
    frozen = {str(code / relative): value for relative, value in code_files.items()}
    frozen.update({row['path']: row['sha256'] for row in chain})
    frozen.update({str(data): file_sha(data), str(initial): file_sha(initial),
                   str(Path(__file__).resolve()): file_sha(__file__)})
    for expected, actual, name in ((args.expected_initial_sha, frozen[str(initial)], 'initial'),
                                   (args.expected_data_sha, frozen[str(data)], 'data')):
        if expected is not None:
            require(digest(expected, name) == actual, name + ' SHA与显式门槛不符')
    refs = dict(format='boundary_relation_pair_reference_v1', created_utc=utc(),
                code_root=str(code), code_files=code_files,
                queue_source=dict(path=str(Path(__file__).resolve()), sha256=file_sha(__file__)),
                config=dict(path=str(config), sha256=file_sha(config), inheritance=chain,
                            resolved_sha256=canonical_sha(resolved)),
                data_manifest=dict(path=str(data), sha256=frozen[str(data)]),
                initial=dict(path=str(initial), sha256=frozen[str(initial)]),
                output=str(output), output_argument=str(args.output), device=args.device,
                max_updates=target, schedule_updates=schedule, batch_size=batch,
                learning_rate=start_lr, minimum_learning_rate=end_lr,
                monitor_updates=cfg['monitor_updates'], checkpoint_updates=cfg['checkpoint_updates'],
                samples_per_epoch=len(labels), steps_per_epoch=math.ceil(len(labels)/batch),
                no_initial_creation=True, no_resume=True, no_full_inference=True,
                no_automatic_merges=True, references_only_no_data_or_weight_copy=True)
    return refs, frozen, labels, monitor_labels


def check_frozen(frozen):
    for path, expected in frozen.items():
        require(file_sha(path) == expected, '冻结输入或源码改变：' + path)


def expected_learning_rate(step, refs):
    """独立核实际第step次更新所用的Python标量cosine学习率。"""
    ratio = (step-1) / max(refs['schedule_updates']-1, 1)
    return refs['minimum_learning_rate'] + .5 * (refs['learning_rate']-refs['minimum_learning_rate']) * (1+math.cos(math.pi*ratio))


def audit_arm(output, mode, refs, labels, monitor_labels):
    base = Path(output) / mode
    done, identity, init = [read_json(base / name) for name in
                            ('complete.json', 'identity.json', 'initialization.json')]
    n, target, spe = len(labels), refs['max_updates'], refs['steps_per_epoch']
    require(done.get('complete') is True and done.get('mode') == mode and
            done.get('optimizer_updates') == target and done.get('global_step') == target and
            done.get('target_updates') == target and done.get('schedule_updates') == refs['schedule_updates'],
            mode + ' 未完整达到配置的更新终点')
    require(done.get('steps_per_epoch') == spe and done.get('samples_per_epoch') == n and
            done.get('completed_epochs') == target // spe and
            done.get('last_epoch') == (target-1)//spe+1 and
            done.get('last_batch_in_epoch') == (target-1) % spe+1, mode + ' epoch/批游标不符')
    require(done.get('original_instances_unmodified') is True and done.get('no_C_weights_loaded') is True,
            mode + ' 必须保持主线和原实例不变')
    require(identity.get('mode') == mode and done.get('identity_sha256') == canonical_sha(identity),
            mode + ' identity回执不符')
    require(identity.get('source_sha256') == refs['code_files'] and
            identity.get('data_manifest_sha256') == refs['data_manifest']['sha256'] and
            identity.get('initial_checkpoint_sha256') == refs['initial']['sha256'],
            mode + ' 代码/数据/初态引用不符')
    require(identity.get('no_automatic_merges') is True and identity.get('no_GT_features') is True and
            identity.get('no_C_weights_loaded') is True, mode + ' 输入隔离合同不符')
    initial_sha = digest(identity.get('initial_state_sha256'), 'initial_state_sha256')
    require(init.get('mode') == mode and init.get('random_initialization') is True and
            init.get('external_trained_weights_inherited') is False and init.get('resume') is False and
            init.get('C_networks_loaded') is False and init.get('initial_state_sha256') == initial_sha and
            init.get('initial_checkpoint_sha256') == refs['initial']['sha256'] and
            done.get('initial_state_sha256') == initial_sha, mode + ' 不是共同随机e0的全新训练')
    rows = [json.loads(line) for line in (base/'steps.jsonl').read_text(encoding='utf8').splitlines() if line.strip()]
    require(len(rows) == target, mode + ' 日志必须覆盖全部更新，不能只有续训后缀')
    for step, row in enumerate(rows, 1):
        require(all(key in row for key in STEP_FIELDS), mode + ' 缺少实际配对日志字段')
        epoch, batch = (step-1)//spe+1, (step-1) % spe+1
        expected_n = min(refs['batch_size'], n-(batch-1)*refs['batch_size'])
        ids = row['sample_ids']
        require(row['global_step'] == step and row['epoch'] == epoch and row['batch_in_epoch'] == batch and
                len(ids) == row['samples'] == expected_n and len(set(ids)) == len(ids) and
                all(sid in labels for sid in ids), mode + ' 更新/样本游标不符')
        require(row['keep'] == sum(labels[sid] == 1 for sid in ids) and
                row['merge'] == sum(labels[sid] == 0 for sid in ids), mode + ' 实际标签计数不符')
        require(isinstance(row['learning_rate'], (int, float)) and math.isfinite(row['learning_rate']) and
                row['learning_rate'] > 0, mode + ' 学习率无效')
        require(math.isclose(row['learning_rate'], expected_learning_rate(step, refs),
                             rel_tol=1e-12, abs_tol=1e-15),
                mode + f' 第{step}步学习率与cosine日程不一致')
        digest(row['shared_draw_sha256'], 'shared_draw_sha256')
    for epoch in range(1, rows[-1]['epoch']+1):
        plan = read_json(base/f'sampling_e{epoch:03d}.json')
        ids = [sid for row in rows if row['epoch'] == epoch for sid in row['sample_ids']]
        require(plan.get('epoch') == epoch and plan.get('no_replacement') is True and
                plan.get('stateless_D4') is True and len(plan['sample_ids']) == n and
                set(plan['sample_ids']) == set(labels) and
                plan.get('order_sha256') == canonical_sha(plan['sample_ids']) and
                ids == plan['sample_ids'][:len(ids)], mode + ' 完整采样计划/实际前缀不符')
    require(done.get('monitor_updates') == refs['monitor_updates'] and
            done.get('checkpoint_updates') == refs['checkpoint_updates'], mode + ' 保存步合同不符')
    observed = {p.name for p in (base/'monitor').iterdir() if p.is_dir()}
    require(observed == {f's{step:06d}' for step in refs['monitor_updates']}, mode + ' monitor目录缺失或混入旧轮数输出')
    for step in refs['monitor_updates']:
        receipt = read_json(base/'monitor'/f's{step:06d}'/'receipt.json')
        require(receipt.get('complete') is True and receipt.get('global_step') == step and
                receipt.get('mode') == mode, mode + ' monitor回执不完整')
        r = receipt.get('rows', [])
        require(len(r) == len(monitor_labels) and len({x['sample_id'] for x in r}) == len(r) and
                {x['sample_id']: x.get('label') for x in r} == monitor_labels,
                mode + ' monitor样本或未知标签隔离不符')
        model_sha = digest(receipt.get('model_sha256'), 'monitor.model_sha256')
        if step in (0, target):
            require(model_sha == (initial_sha if step == 0 else done.get('model_sha256')),
                    mode + ' monitor不属于对应的初/终态')
    checkpoint_files = {}
    for name in ['last.pt'] + [f'u{step:06d}.pt' for step in refs['checkpoint_updates']]:
        path = base/name
        require(path.is_file() and path.stat().st_size > 0, mode + ' 缺少续训/阶段权重：' + name)
        checkpoint_files[name] = dict(path=str(path), bytes=path.stat().st_size, sha256=file_sha(path))
    return dict(identity=identity, completion=done, rows=rows, checkpoint_files=checkpoint_files,
                samples_presented=sum(row['samples'] for row in rows))


def audit_pair(arms, refs):
    left, right = arms['maps'], arms['rgb']
    a, b = [dict(v['identity']) for v in (left, right)]
    a.pop('mode'); b.pop('mode')
    require(canonical(a) == canonical(b), '两arm身份除mode外不完全相同')
    for step, (a, b) in enumerate(zip(left['rows'], right['rows']), 1):
        require(all(a[key] == b[key] for key in STEP_FIELDS), f'第{step}步实际采样/D4/标签/LR不一致')
    return dict(complete=True, completed_utc=utc(), max_updates=refs['max_updates'],
                updates_each=refs['max_updates'], samples_presented_each=left['samples_presented'],
                same_identity_except_mode=True, same_initial_checkpoint=True,
                same_actual_sample_order=True, same_actual_post_D4_maps_masks_labels_SHA=True,
                same_learning_rates_and_label_counts=True,
                D4_evidence='Frozen trainer shared_draw_sha256 covers actual post-D4 maps, masks and ordered labels; no model replay.',
                initial_state_sha256=left['identity']['initial_state_sha256'],
                monitor_updates=refs['monitor_updates'], no_full_inference=True,
                no_automatic_merges=True, no_accuracy_or_convergence_claim=True,
                arms={mode: {k: value[k] for k in ('completion', 'checkpoint_files', 'samples_presented')}
                      for mode, value in arms.items()})


def run_queue(args, *, popen_factory=None):
    refs, frozen, labels, monitor_labels = preflight(args)
    output = Path(refs['output'])
    output.mkdir(parents=True, exist_ok=True)
    lock = output/'.relation_pair.lock'
    with lock.open('x', encoding='utf8') as stream:
        stream.write(canonical(dict(pid=os.getpid(), started_utc=utc())))
    state = dict(format='boundary_relation_pair_queue_v1', pid=os.getpid(), started_utc=utc(),
                 state='preflight', completed_modes=[], command_history=[], max_updates=refs['max_updates'],
                 no_full_inference=True, no_automatic_merges=True, output=refs['output'])
    child, arms, owns_run = None, {}, False
    try:
        for name in RESERVED:
            require(not os.path.lexists(output/name), '拒绝覆盖已有正式产物：' + str(output/name))
        write_json(output/'queue_reference.json', refs)
        owns_run = True
        for mode in ('maps', 'rgb'):
            check_frozen(frozen)
            argv = [sys.executable, str(Path(refs['code_root'])/'train_boundary_relation.py'),
                    '--config', refs['config']['path'], '--mode', mode,
                    '--data-manifest', refs['data_manifest']['path'], '--output', str(output/mode),
                    '--initial-load', refs['initial']['path'], '--device', refs['device']]
            state.update(state='starting', mode=mode, argv=argv, mode_started_utc=utc(), child_pid=None)
            state['command_history'].append(dict(mode=mode, argv=argv, started_utc=state['mode_started_utc'], pid=None))
            write_json(output/'queue_status.json', state)
            env = dict(os.environ, PYTHONUNBUFFERED='1')
            kwargs = dict(cwd=refs['code_root'], env=env, stderr=subprocess.STDOUT)
            if os.name == 'nt':
                kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
            with (output/f'{mode}.log').open('xb') as log:
                child = (popen_factory or subprocess.Popen)(argv, stdout=log, **kwargs)
                state.update(state='running', child_pid=child.pid)
                state['command_history'][-1]['pid'] = child.pid
                write_json(output/'queue_status.json', state)
                code = child.wait()
            state.update(returncode=code, mode_finished_utc=utc(), child_pid=None)
            state['command_history'][-1].update(returncode=code, finished_utc=state['mode_finished_utc'])
            require(code == 0, mode + ' trainer退出失败，停止后续arm：' + str(code))
            state['state'] = 'auditing_arm'
            write_json(output/'queue_status.json', state)
            check_frozen(frozen)
            arms[mode] = audit_arm(output, mode, refs, labels, monitor_labels)
            state['completed_modes'].append(mode)
        state['state'] = 'auditing_pair'
        write_json(output/'queue_status.json', state)
        contract = audit_pair(arms, refs)
        check_frozen(frozen)
        write_json(output/'paired_contract.json', contract)
        state.update(state='complete', completed_utc=utc(), paired_contract_sha256=file_sha(output/'paired_contract.json'))
        write_json(output/'queue_status.json', state)
        return contract
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        state.update(state='failed', failed_utc=utc(), error_type=type(error).__name__, error=str(error))
        if owns_run:
            write_json(output/'queue_status.json', state)
        raise
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-root', default=str(ROOT))
    parser.add_argument('--config', required=True)
    parser.add_argument('--data-manifest', required=True)
    parser.add_argument('--initial-load', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--expected-initial-sha')
    parser.add_argument('--expected-data-sha')
    args = parser.parse_args()
    contract = run_queue(args)
    print(json.dumps(dict(complete=True, output=str(Path(args.output).resolve()),
                          updates_each=contract['updates_each'],
                          samples_presented_each=contract['samples_presented_each']), ensure_ascii=False))


if __name__ == '__main__':
    main()
