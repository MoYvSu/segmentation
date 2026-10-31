# -*- coding: utf-8 -*-
"""只验证阶段编号与固定索引，不读取图像或生成评分。"""
from dataclasses import FrozenInstanceError

import pytest

from utils.submission_stage import dataset_contract


def test_default_contract_retains_historical_100_images():
    contract = dataset_contract()
    assert contract == dataset_contract('quarter_final_100')
    assert (contract.name, contract.first, contract.last, contract.count) == ('quarter_final_100', 69, 168, 100)
    assert contract.names == tuple(f'test_{i:03d}' for i in range(69, 169))
    assert contract.source_index('test_069') == 0
    assert contract.source_index('test_101') == 32
    assert contract.source_index('test_168') == 99


def test_semifinal_contract_is_explicit_150_and_subset_order_independent():
    contract = dataset_contract('semifinal_150')
    assert (contract.first, contract.last, contract.count) == (169, 318, 150)
    assert len(set(contract.names)) == 150
    subset = ['test_318', 'test_201', 'test_169']
    assert [contract.source_index(stem) for stem in subset] == [149, 32, 0]
    assert contract.names[0] == 'test_169' and contract.names[-1] == 'test_318'
    with pytest.raises(FrozenInstanceError):
        contract.first = 0


@pytest.mark.parametrize('name', ['', 'semifinal', 'SEMIFINAL_150', 'quarter_final_150', False, 150, {}])
def test_unknown_stage_fails_without_guessing(name):
    with pytest.raises(ValueError, match='Unknown submission dataset'):
        dataset_contract(name)


@pytest.mark.parametrize('stem', [
    'test_0169', 'test_69', 'test_１６９', 'test_١٦٩', 'test_169.png',
    'test_169\n', ' test_169', 'test_169 ', '../test_169', 'data/test_169',
    r'data\test_169', 'C:/test_169', None, 169,
])
def test_stems_are_three_ascii_digits_without_paths_or_extensions(stem):
    with pytest.raises(ValueError, match='three ASCII digits'):
        dataset_contract('semifinal_150').source_index(stem)


@pytest.mark.parametrize('stage,stem', [
    ('quarter_final_100', 'test_068'), ('quarter_final_100', 'test_169'),
    ('semifinal_150', 'test_168'), ('semifinal_150', 'test_319'),
    ('semifinal_150', 'test_069'),
])
def test_cross_stage_number_cannot_be_reused(stage, stem):
    with pytest.raises(ValueError, match='requires test_'):
        dataset_contract(stage).source_index(stem)
