# -*- coding: utf-8 -*-
"""相邻P方案改为独立留档脚本，共享NAF入口不保留实验开关。"""
import builtins
import json

import pytest

from tools import infer_boundary_naf as wrapper
from test_infer_boundary_naf import install_fake_entry, one_cpu_thread


def test_shared_entry_has_no_plain_P_source_or_apply_hook():
    assert not hasattr(wrapper, '_apply_pearlite_result')
    assert not any('pearlite_merge' in name for name in wrapper._sources())


def test_shared_entry_does_not_import_P_merge_or_change_parameter_account(tmp_path, monkeypatch):
    args, state, output, naf, base_count, *_ = install_fake_entry(tmp_path, monkeypatch)
    original_import = builtins.__import__

    def guarded(name, *values, **options):
        if name == 'utils.pearlite_merge':
            pytest.fail('共享NAF入口不得导入已独立留档的P合并模块')
        return original_import(name, *values, **options)

    monkeypatch.setattr(builtins, '__import__', guarded)
    receipt = wrapper.run(args, device='cpu')
    assert receipt['parameter_count'] == base_count + sum(p.numel() for p in naf.parameters())
    assert 'pearlite_merge' not in receipt and not (output/'audit/pearlite').exists()
    assert not any('pearlite_merge' in name for name in receipt['imported_sources'])
    report = json.loads((output/'audit/report.json').read_text(encoding='utf8'))
    assert 'pearlite_merge' not in report and 'pearlite_merge' not in report['models']
    assert state['raw_predict_calls'] == 1


def test_removed_CLI_flag_is_rejected_before_running_models(monkeypatch):
    monkeypatch.setattr(wrapper, 'run', lambda *_: pytest.fail('旧实验开关不应触发共享入口'))
    with pytest.raises(SystemExit) as error:
        wrapper.main(['--input-dir', 'i', '--output-dir', 'o', '--naf-checkpoint', 'n',
                      '--merge-adjacent-pearlite'])
    assert error.value.code == 2
