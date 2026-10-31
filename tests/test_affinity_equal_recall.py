# -*- coding: utf-8 -*-
"""合成监督域与等召回诊断验证；不使用赛题产物。"""
import numpy as np
import ast
import inspect
import json
from types import SimpleNamespace
import pytest
from tools.probe_affinity_equal_recall import pair_targets, fused_targets, score_histogram, equal_recall, curve_metrics, select_checkpoints, run


def test_actual_report_serializes_after_channel_slice_loop():
    """执行实际checkpoint赋值、offset循环及报告构建，省去GPU前向。"""
    tree=ast.parse(inspect.getsource(run))
    checkpoint_assignment=next(node for node in ast.walk(tree) if isinstance(node,ast.Assign)
                               and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name)
                               and node.value.func.id=='select_checkpoints')
    channel_loop=next(node for node in ast.walk(tree) if isinstance(node,ast.For)
                      and isinstance(node.target,ast.Tuple) and any(isinstance(n,ast.Name) and n.id=='layer' for n in node.target.elts))
    report_assignment=next(node for node in ast.walk(tree) if isinstance(node,ast.Assign)
                           and any(isinstance(target,ast.Name) and target.id=='report' for target in node.targets))
    args=SimpleNamespace(control_checkpoint='native.pt',control_sha256='a'*64,
                         balance_checkpoint='mixed.pt',balance_sha256='b'*64,seed=20261001,bins=8192)
    context=dict(select_checkpoints=select_checkpoints,args=args,
                 config=dict(backend_adaptation=dict(restoration={})),
                 cases=[],records=[],checkpoints={},mode='gated',kwargs={},pooled={},paired=[],receipts={},
                 started=0,time=SimpleNamespace(time=lambda:0),
                 torch=SimpleNamespace(cuda=SimpleNamespace(max_memory_allocated=lambda:0)),
                 sha=lambda path:'a'*64,__file__='synthetic.py')
    # 仅删除循环中的GPU/计数工作，保留生产代码的迭代器和循环变量赋值。
    channel_loop.body=[ast.Pass()]
    module=ast.fix_missing_locations(ast.Module(body=[checkpoint_assignment,channel_loop,report_assignment],type_ignores=[]))
    exec(compile(module,'actual_report_without_gpu','exec'),context)
    serialized=json.dumps(context['report'],allow_nan=False)
    assert json.loads(serialized)['comparison_selection']=='explicit_pair'


def test_default_checkpoint_selection_remains_native_control_and_native_balance():
    config = dict(paths=dict(project_root='.'),
                  affinity_native=dict(penalty_reference='outputs/affinity_native/p1024', penalty_sha256='a'*64),
                  backend_adaptation=dict(output_dir='outputs/affinity_balance/native'))
    calls=[]
    def status_reader(path):
        calls.append(path)
        return dict(final_checkpoint_sha256='b'*64)
    paths, expected, selection=select_checkpoints(config,SimpleNamespace(),status_reader)
    assert paths==dict(control='outputs/affinity_native/p1024/final.pt', balance='outputs/affinity_balance/native/final.pt')
    assert expected==dict(control='a'*64,balance='b'*64)
    assert selection=='default_old_native_vs_native_balance' and len(calls)==1


def test_explicit_pair_bypasses_default_status_and_records_actual_paths():
    args=SimpleNamespace(control_checkpoint='outputs/affinity_balance/native/final.pt',control_sha256='A'*64,
                         balance_checkpoint='outputs/affinity_balance/mixed/final.pt',balance_sha256='B'*64)
    def forbidden_status(path):
        raise AssertionError('Explicit pair must not read old default status')
    paths,expected,selection=select_checkpoints({},args,forbidden_status)
    assert paths==dict(control=args.control_checkpoint,balance=args.balance_checkpoint)
    assert expected==dict(control='a'*64,balance='b'*64)
    assert selection=='explicit_pair'


@pytest.mark.parametrize('options',[dict(control_checkpoint='some.pt'),
                                  dict(control_checkpoint='a.pt',control_sha256='a'*64,balance_checkpoint='b.pt'),
                                  dict(control_checkpoint='',control_sha256='a'*64,balance_checkpoint='b.pt',balance_sha256='b'*64)])
def test_partial_checkpoint_override_is_rejected(options):
    with pytest.raises(ValueError,match='both checkpoint paths and both SHA256'):
        select_checkpoints({},SimpleNamespace(**options))


@pytest.mark.parametrize('bad_sha',['a'*63,'g'*64,'a'*64+' ',''])
def test_invalid_explicit_hash_is_rejected(bad_sha):
    args=SimpleNamespace(control_checkpoint='a.pt',control_sha256=bad_sha,balance_checkpoint='b.pt',balance_sha256='b'*64)
    with pytest.raises(ValueError):
        select_checkpoints({},args)


def test_unknown_endpoints_and_padding_are_ignored():
    ids = np.array([[1, 1, 2, 2], [1, 1, 2, 2]], np.int32)
    trusted = np.ones_like(ids, bool); trusted[:, 2] = False
    target, valid, _ = pair_targets(ids, trusted, grid=4)
    assert not valid[:, 2:].any()
    assert not valid[0, :2, 1:3].any()
    assert valid[0, :2, 0].all() and target[0, :2, 0].all()
    assert not ((~target[:4]) & valid[:4]).any()
    # 长offset允许跨过未知像素，但两个端点必须可信；这是既有关系契约。
    assert valid[4, 0, 1] and not target[4, 0, 1]


def test_fused_boundary_requires_all_short_relations_known():
    target = np.ones((8, 2, 2), bool); valid = np.ones_like(target)
    target[1, 0, 0] = False; valid[3, 1, 1] = False
    boundary, eligible = fused_targets(target, valid)
    assert boundary[0, 0] and eligible[0, 0]
    assert not eligible[1, 1] and not boundary[1, 1]
    assert eligible[0, 1] and not boundary[0, 1]


def test_strict_fusion_interior_excludes_long_range_real_boundaries():
    target=np.ones((8,2,2),bool);valid=np.ones_like(target)
    target[4,0,0]=False
    target[1,0,1]=False
    valid[7,1,0]=False
    boundary,eligible=fused_targets(target,valid,strict_interior=True)
    assert not boundary[0,0] and not eligible[0,0]
    assert boundary[0,1] and eligible[0,1]
    assert not eligible[1,0]
    assert eligible[1,1] and not boundary[1,1]


def test_equal_recall_is_invariant_to_monotonic_calibration_on_discrete_example():
    scores = np.array([.95, .85, .75, .65, .55, .45, .35, .25])
    truth = np.array([True, True, False, True, False, False, True, False])
    valid = np.ones_like(truth)
    old = score_histogram(scores, truth, valid, bins=1024)
    calibrated = score_histogram(scores * .6, truth, valid, bins=1024)
    for recall in (.5, .75, 1):
        a, b = equal_recall(old, recall), equal_recall(calibrated, recall)
        assert a['interior_false_boundary_rate'] == b['interior_false_boundary_rate']
        assert a['attained_true_boundary_recall'] == b['attained_true_boundary_recall']
    assert abs(curve_metrics(old)['roc_auc'] - curve_metrics(calibrated)['roc_auc']) < 1e-12


def test_improved_ranking_lowers_false_boundaries_at_equal_recall():
    truth = np.array([True, True, False, False]); valid = np.ones(4, bool)
    old = score_histogram(np.array([.9, .4, .8, .3]), truth, valid, bins=128)
    new = score_histogram(np.array([.9, .8, .4, .3]), truth, valid, bins=128)
    assert equal_recall(old, 1)['interior_false_boundary_rate'] == .5
    assert equal_recall(new, 1)['interior_false_boundary_rate'] == 0
