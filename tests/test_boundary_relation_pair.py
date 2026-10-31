# -*- coding: utf-8 -*-
"""配对队列的只读身份、失败停止和1000步预算合同；不运行真实训练。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_relation_pair_tests', ROOT/'tools/train_boundary_relation_pair.py')
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)


def make_run(tmp_path, *, target=4, count=5, batch=2, monitors=None):
    code = tmp_path/'code'
    for name in queue.CODE_FILES:
        path = code/name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name == 'utils/config.py':
            shutil.copyfile(ROOT/name, path)
        else:
            path.write_text('# 合成队列测试，不执行模型\n', encoding='utf8')
    monitor_steps = monitors or [0, 1, target]
    base = code/'base.yaml'
    base.write_text('boundary_relation:\n  batch_size: '+str(batch)+
                   '\n  apply_predicted_merges: false\n  learning_rate: 0.0003\n  minimum_learning_rate: 0.00003\n', encoding='utf8')
    cfg = code/'experiment.yaml'
    cfg.write_text('_base: base.yaml\nboundary_relation:\n  max_updates: '+str(target)+
                   '\n  schedule_updates: '+str(target)+'\n  monitor_updates: '+json.dumps(monitor_steps)+
                   '\n  checkpoint_updates: '+json.dumps(monitor_steps)+'\n', encoding='utf8')
    samples = [dict(sample_id=f'train_001:{i}', source='train_001', label=i % 2) for i in range(count)]
    data = code/'data.json'
    queue.write_json(data, dict(format='boundary_relation_data_v1', complete=True,
        shards=[dict(path='not_read_by_queue.npz', count=count, samples=samples)],
        monitor=dict(train_sample_ids=[samples[0]['sample_id'], samples[1]['sample_id']]),
        monitor_shards=[dict(path='also_not_read_by_queue.npz', count=1,
                            samples=[dict(sample_id='test_169:monitor', source='test_169', label=255)])],
        training_eligibility=dict(eligible_by_count_only=True)))
    initial = code/'initial.pt'
    initial.write_bytes(b'fake-e0-reference-only')
    output = tmp_path/'long1000'
    output.mkdir()
    (output/'code').mkdir()
    (output/'preflight').mkdir()
    (output/'index.html').write_text('用户预先准备的页面', encoding='utf8')
    (output/'queue.log').write_text('外层nohup日志', encoding='utf8')
    return argparse.Namespace(code_root=str(code), config='experiment.yaml', data_manifest='data.json',
        initial_load='initial.pt', output=str(output), device='cuda',
        expected_initial_sha=queue.file_sha(initial), expected_data_sha=queue.file_sha(data))


def emit_arm(mode, parent):
    refs = queue.read_json(parent/'queue_reference.json')
    data = queue.read_json(refs['data_manifest']['path'])
    labels = {r['sample_id']:r['label'] for s in data['shards'] for r in s['samples']}
    ids = list(labels)
    n, batch, target, spe = len(ids), refs['batch_size'], refs['max_updates'], refs['steps_per_epoch']
    base = parent/mode
    base.mkdir()
    initial = 'a'*64
    final = ('b' if mode == 'maps' else 'c')*64
    identity = dict(mode=mode, architecture={'name':'synthetic'},
        recipe=dict(max_updates=target, schedule_updates=refs['schedule_updates']),
        update_protocol=dict(monitor_updates=refs['monitor_updates'], checkpoint_updates=refs['checkpoint_updates']),
        source_sha256=refs['code_files'], data_manifest_sha256=refs['data_manifest']['sha256'],
        data_identity_sha256='d'*64, initial_checkpoint_sha256=refs['initial']['sha256'],
        initial_state_sha256=initial, no_automatic_merges=True, no_GT_features=True, no_C_weights_loaded=True)
    queue.write_json(base/'identity.json', identity)
    queue.write_json(base/'initialization.json', dict(mode=mode, random_initialization=True,
        external_trained_weights_inherited=False, resume=False, C_networks_loaded=False,
        initial_state_sha256=initial, initial_checkpoint_sha256=refs['initial']['sha256']))
    plans = {}
    for epoch in range(1, (target-1)//spe+2):
        shift=(epoch-1) % n
        order=ids[shift:]+ids[:shift]
        plans[epoch]=order
        queue.write_json(base/f'sampling_e{epoch:03d}.json', dict(epoch=epoch, no_replacement=True,
            stateless_D4=True, sample_ids=order, order_sha256=queue.canonical_sha(order)))
    rows=[]
    for step in range(1, target+1):
        epoch, bi=(step-1)//spe+1, (step-1) % spe+1
        selected=plans[epoch][(bi-1)*batch:bi*batch]
        rows.append(dict(global_step=step, epoch=epoch, batch_in_epoch=bi, sample_ids=selected,
            samples=len(selected), keep=sum(labels[s]==1 for s in selected), merge=sum(labels[s]==0 for s in selected),
            learning_rate=queue.expected_learning_rate(step, refs), shared_draw_sha256=queue.canonical_sha([step, selected])))
    (base/'steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf8')
    monitor_labels={sid:labels[sid] for sid in data['monitor']['train_sample_ids']}
    monitor_labels.update({r['sample_id']:None for s in data['monitor_shards'] for r in s['samples']})
    for step in refs['monitor_updates']:
        directory=base/'monitor'/f's{step:06d}'
        directory.mkdir(parents=True)
        queue.write_json(directory/'receipt.json', dict(complete=True, global_step=step, mode=mode,
            model_sha256=initial if step==0 else final,
            rows=[dict(sample_id=sid,label=label) for sid,label in monitor_labels.items()]))
    for name in ['last.pt']+[f'u{s:06d}.pt' for s in refs['checkpoint_updates']]:
        (base/name).write_bytes(b'synthetic-checkpoint')
    queue.write_json(base/'complete.json', dict(complete=True, mode=mode,
        optimizer_updates=target, global_step=target, target_updates=target, schedule_updates=refs['schedule_updates'],
        steps_per_epoch=spe, samples_per_epoch=n, completed_epochs=target//spe,
        last_epoch=(target-1)//spe+1, last_batch_in_epoch=(target-1) % spe+1,
        original_instances_unmodified=True, no_C_weights_loaded=True,
        identity_sha256=queue.canonical_sha(identity), initial_state_sha256=initial, model_sha256=final,
        monitor_updates=refs['monitor_updates'], checkpoint_updates=refs['checkpoint_updates']))


def fake_launcher(calls, *, fail_mode=None, mutate=None):
    def launch(argv, **kwargs):
        mode=argv[argv.index('--mode')+1]
        parent=Path(argv[argv.index('--output')+1]).parent
        calls.append((mode, argv, kwargs))
        emit_arm(mode, parent)
        if mutate:
            mutate(mode, parent)

        class Process:
            pid = 9000+len(calls)
            def wait(self, timeout=None):
                return 7 if mode==fail_mode else 0
            def poll(self):
                return 7 if mode==fail_mode else 0
        return Process()
    return launch


def test_sequential_pair_keeps_existing_root_and_reference_only(tmp_path):
    args=make_run(tmp_path)
    calls=[]
    answer=queue.run_queue(args,popen_factory=fake_launcher(calls))
    assert [x[0] for x in calls]==['maps','rgb']
    assert answer['samples_presented_each']==7 and answer['updates_each']==4
    assert answer['same_identity_except_mode'] and answer['no_automatic_merges']
    for mode,argv,kwargs in calls:
        assert '--initial-create' not in argv and '--resume' not in argv and '--smoke' not in argv
        assert argv[argv.index('--initial-load')+1]==str(Path(args.code_root)/'initial.pt')
        assert kwargs['cwd']==args.code_root
    parent=Path(args.output)
    assert (parent/'queue.log').read_text(encoding='utf8')=='外层nohup日志'
    assert (parent/'index.html').read_text(encoding='utf8')=='用户预先准备的页面'
    assert queue.read_json(parent/'queue_status.json')['state']=='complete'
    history=queue.read_json(parent/'queue_status.json')['command_history']
    assert [r['mode'] for r in history]==['maps','rgb']
    assert all(r['pid'] and r['returncode']==0 and r['finished_utc'] for r in history)
    assert not (parent/'.relation_pair.lock').exists()


def test_1000_updates_are_23_full_epochs_and_11_batches(tmp_path):
    args=make_run(tmp_path,target=1000,count=2746,batch=64,monitors=[0,215,500,750,1000])
    answer=queue.run_queue(args,popen_factory=fake_launcher([]))
    assert answer['updates_each']==1000 and answer['samples_presented_each']==63862
    done=answer['arms']['maps']['completion']
    assert (done['completed_epochs'],done['last_epoch'],done['last_batch_in_epoch'])==(23,24,11)
    assert answer['monitor_updates']==[0,215,500,750,1000]
    steps=[json.loads(x) for x in (Path(args.output)/'maps/steps.jsonl').read_text().splitlines()]
    assert steps[0]['learning_rate']==.0003 and steps[-1]['learning_rate']==.00003


@pytest.mark.parametrize('reserved',['maps','rgb','queue_status.json','paired_contract.json','queue_contract.json'])
def test_existing_formal_artifact_is_never_overwritten(tmp_path,reserved):
    args=make_run(tmp_path)
    path=Path(args.output)/reserved
    path.write_text('不可覆盖',encoding='utf8')
    with pytest.raises(ValueError,match='拒绝覆盖'):
        queue.run_queue(args,popen_factory=lambda *a,**k:pytest.fail('不得启动子进程'))
    assert path.read_text(encoding='utf8')=='不可覆盖'


@pytest.mark.parametrize('field',['expected_initial_sha','expected_data_sha'])
def test_explicit_hash_gate_fails_before_launch(tmp_path,field):
    args=make_run(tmp_path)
    setattr(args,field,'0'*64)
    with pytest.raises(ValueError,match='SHA'):
        queue.run_queue(args,popen_factory=lambda *a,**k:pytest.fail('不得启动子进程'))
    assert not (Path(args.output)/'queue_status.json').exists()


def test_failed_first_trainer_halts_second_arm(tmp_path):
    args=make_run(tmp_path);calls=[]
    with pytest.raises(ValueError,match='trainer退出失败'):
        queue.run_queue(args,popen_factory=fake_launcher(calls,fail_mode='maps'))
    assert [x[0] for x in calls]==['maps']
    assert not (Path(args.output)/'rgb').exists()
    assert queue.read_json(Path(args.output)/'queue_status.json')['state']=='failed'


@pytest.mark.parametrize('bad',['missing_monitor','suffix_log','bad_initial'])
def test_bad_first_arm_receipt_halts_second(tmp_path,bad):
    args=make_run(tmp_path);calls=[]
    def mutate(mode,parent):
        if bad=='missing_monitor':
            shutil.rmtree(parent/mode/'monitor'/'s000001')
        elif bad=='suffix_log':
            path=parent/mode/'steps.jsonl';path.write_text('\n'.join(path.read_text().splitlines()[1:])+'\n')
        else:
            path=parent/mode/'initialization.json';r=queue.read_json(path);r['resume']=True;queue.write_json(path,r)
    with pytest.raises(ValueError):
        queue.run_queue(args,popen_factory=fake_launcher(calls,mutate=mutate))
    assert [x[0] for x in calls]==['maps']


@pytest.mark.parametrize('bad',['shared_draw_sha256','learning_rate','identity'])
def test_cross_arm_actual_pairing_mismatch_fails(tmp_path,bad):
    args=make_run(tmp_path)
    def mutate(mode,parent):
        if mode!='rgb':return
        if bad=='identity':
            path=parent/mode/'identity.json';identity=queue.read_json(path);identity['data_identity_sha256']='e'*64;queue.write_json(path,identity)
            done_path=parent/mode/'complete.json';done=queue.read_json(done_path);done['identity_sha256']=queue.canonical_sha(identity);queue.write_json(done_path,done)
        else:
            path=parent/mode/'steps.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()]
            rows[1][bad]='e'*64 if bad=='shared_draw_sha256' else .9
            path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf8')
    with pytest.raises(ValueError,match='不一致|不完全相同'):
        queue.run_queue(args,popen_factory=fake_launcher([],mutate=mutate))
    assert not (Path(args.output)/'paired_contract.json').exists()


def test_source_mutation_between_arms_halts_queue(tmp_path):
    args=make_run(tmp_path);calls=[]
    def mutate(mode,parent):
        (Path(args.code_root)/'base.yaml').write_text('被修改',encoding='utf8')
    with pytest.raises(ValueError,match='冻结输入或源码改变'):
        queue.run_queue(args,popen_factory=fake_launcher(calls,mutate=mutate))
    assert [x[0] for x in calls]==['maps']


def test_both_arms_cannot_pass_with_the_same_wrong_learning_rate(tmp_path):
    args=make_run(tmp_path);calls=[]
    def mutate(mode,parent):
        path=parent/mode/'steps.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()]
        rows[0]['learning_rate']=.0002
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf8')
    with pytest.raises(ValueError,match='cosine日程不一致'):
        queue.run_queue(args,popen_factory=fake_launcher(calls,mutate=mutate))
    assert [r[0] for r in calls]==['maps']
