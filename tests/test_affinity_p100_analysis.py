# -*- coding: utf-8 -*-
"""P100分析门禁与统计复用隔离；不需要赛题图像或GPU。"""
from copy import deepcopy
import types

import pytest

from tools import analyze_affinity_p100 as analysis
from utils.config import load_config


def pipeline():
    config = {'sample': True}
    return config, dict(status='complete', stage='done', smoke=False,
        control_retrained=False, completed=['p100', 'p100_infer'], config=config)


@pytest.mark.parametrize('key,value', [('status', 'running'), ('stage', 'p100_infer'),
    ('completed', ['p100']), ('completed', ['p025', 'p025_infer']),
    ('smoke', True), ('control_retrained', True), ('config', {})])
def test_no_partial_or_other_candidate_is_analyzable(key, value):
    config, complete = pipeline()
    analysis.require_complete_pipeline(complete, config)
    with pytest.raises(RuntimeError, match='Complete isolated P100'):
        analysis.require_complete_pipeline(dict(complete, **{key: value}), config)


def manifest():
    return dict(complete=True, epoch=20, precision='FP32', checkpoint_sha256='fixed-e20',
        sizes=[1024], images=[dict(source=f'test_{index:03}', view='patch1024') for index in range(100)])


@pytest.mark.parametrize('key,value', [('complete', False), ('epoch', 16), ('precision', 'FP16'),
    ('checkpoint_sha256', 'another-head'), ('sizes', [512]), ('sizes', [512, 1024])])
def test_final_output_requires_exact_e20_native_manifest(key, value):
    complete = manifest()
    assert len(analysis.require_manifest_record(complete, 'fixed-e20', candidate=True)) == 100
    with pytest.raises(RuntimeError, match='deployment manifest'):
        analysis.require_manifest_record(dict(complete, **{key: value}), 'fixed-e20', candidate=True)


def test_output_manifest_has_exactly_one_hundred_distinct_native_sources():
    complete = manifest()
    for rows in (complete['images'][:-1], complete['images'] + [complete['images'][0]],
                 complete['images'][:-1] + [complete['images'][0]]):
        with pytest.raises(RuntimeError, match='deployment manifest'):
            analysis.require_manifest_record(dict(complete, images=rows), 'fixed-e20', candidate=True)
    # 旧控制可同时保留另一尺度；主比较仍只使用原生1024这100项。
    control = dict(complete, sizes=[512, 1024], images=complete['images'] +
                   [dict(source='extra', view='patch512')])
    assert len(analysis.require_manifest_record(control, 'fixed-e20', candidate=False)) == 100


def test_actual_source_coefficients_are_fixed_and_p025_is_rejected():
    config = load_config('config/train/affinity_source_p100.yaml')
    assert analysis.require_coefficients(config) == dict(beta=1., alpha=.75,
        manual_coefficient=.75, pseudo_coefficient=.75, nominal_coefficient_sum=1.5)
    changed = deepcopy(config)
    changed['backend_adaptation']['pseudo_weight'] = .25
    with pytest.raises((RuntimeError, ValueError)):
        analysis.require_coefficients(changed)


def test_formal_launch_binds_exact_smoke_and_reused_control():
    binding = dict(path='smoke/pipeline_status.json', sha256='gate',
                   control_reuse_sha256='reuse', parity_sha256='parity')
    control = dict(passed=True, control_retrained=False, technical_updates_reused=8)
    formal = dict(smoke_gate=binding, reused_control=control)
    smoke = dict(reused_control=control)
    analysis.require_smoke_binding(formal, smoke, binding)
    for changed in (dict(formal, smoke_gate=None), dict(formal, smoke_gate={'sha256': 'different'}),
                    dict(formal, reused_control={'passed': True, 'other_control': True})):
        with pytest.raises(RuntimeError, match='exact smoke/reused-control'):
            analysis.require_smoke_binding(changed, smoke, binding)


def test_current_and_archived_sources_are_both_bound(monkeypatch, tmp_path):
    monkeypatch.setattr(analysis, 'ROOT', tmp_path)
    monkeypatch.setattr(analysis.runner, 'source_names', lambda: ('registered.py',))
    root = tmp_path / 'run'
    (root / 'source').mkdir(parents=True)
    source = tmp_path / 'registered.py'
    snapshot = root / 'source/registered.py'
    source.write_text('original\n', encoding='utf8')
    snapshot.write_bytes(source.read_bytes())
    sources = {'registered.py': analysis.sha(source)}
    analysis.verify_sources(root, sources)
    snapshot.write_text('changed snapshot\n', encoding='utf8')
    with pytest.raises(RuntimeError, match='archived training source changed'):
        analysis.verify_sources(root, sources)
    snapshot.write_bytes(source.read_bytes())
    source.write_text('changed current\n', encoding='utf8')
    with pytest.raises(RuntimeError, match='archived training source changed'):
        analysis.verify_sources(root, sources)
    with pytest.raises(RuntimeError, match='source closure'):
        analysis.verify_sources(root, {})


def all_constants(code):
    for value in code.co_consts:
        if isinstance(value, types.CodeType):
            yield from all_constants(value)
        elif isinstance(value, tuple):
            yield from value
        else:
            yield value


def test_diagnostic_adapter_changes_labels_only_and_leaves_old_module_immutable():
    originals = {name: getattr(analysis.hard, name) for name in
                 ('load_checked', 'gt_totals', 'probe', 'gt', 'test')}
    before = {name: function.__code__ for name, function in originals.items()}
    namespace = analysis._diagnostics()
    assert namespace['ARMS'] == ('r1', 'p100')
    assert analysis.hard.ARMS == ('r1', 'h0')
    for name, original in originals.items():
        adapted = namespace[name]
        assert original.__code__ is before[name]
        assert adapted is not original and adapted.__globals__ is namespace
        assert adapted.__code__.co_code == original.__code__.co_code
        assert adapted.__code__.co_names == original.__code__.co_names
        assert adapted.__code__.co_varnames == original.__code__.co_varnames
        assert adapted.__defaults__ == original.__defaults__
        numeric = lambda fn: [value for value in all_constants(fn.__code__)
                             if isinstance(value, (int, float))]
        assert numeric(adapted) == numeric(original)
        assert all(not any(label in value for label in ('h0', 'H0', 'p025', 'P025'))
                   for value in all_constants(adapted.__code__) if isinstance(value, str))
    assert analysis._map_constant(('h0', 'H0 r1 h0 e20', (119, 'mixed'))) == (
        'p100', 'P100 r1 h1 beta1 e20', (119, 'mixed'))


def test_reused_synthetic_statistics_preserve_unknown_ignore_and_boundary_direction():
    analysis.self_test()
