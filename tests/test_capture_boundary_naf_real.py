# -*- coding: utf-8 -*-
"""N1捕获保持同输入，拒绝伪零噪声与来源漂移。"""
import pytest

from tools.capture_boundary_naf_real import n1_config, training_sources, capture_n1_window


def test_n1_only_changes_local_policy_without_mutating_n0():
    original = {'submission': {'local_restoration': {'mode': 'single', 'model': 'restorer'},
                                'restoration_noise': 'cuda_philox4090_v1'}, 'fixed': [1, 2]}
    result = n1_config(original)
    assert result['submission']['local_restoration'] == {'mode': 'zero', 'model': 'restorer'}
    assert result['submission']['restoration_noise'] == original['submission']['restoration_noise']
    assert original['submission']['local_restoration']['mode'] == 'single'
    assert result['fixed'] == original['fixed']


def sources():
    row = dict(source='train_001', shape=[64,64], hashes={
        'image_sha256':'a'*64, 'gt_sha256':'b'*64, 'classes_sha256':'c'*64})
    old = dict(source='train_001', image_sha256='a'*64, shape=[64,64], source_index=782,
               pool_match={'sha256':'a'*64, 'full_pool_index':782},
               processed_gt={'sha256':'b'*64, 'classes_sha256':'c'*64})
    return {'rows':[row]}, {'complete':True, 'files':[old]}


def test_source_index_follows_content_not_alias():
    gt,old=sources()
    rows=training_sources(gt,old,1)
    assert rows[0]['source_index']==782
    assert rows[0]['domain']=='train_gt'


@pytest.mark.parametrize('field', ['image_sha256','gt_sha256','classes_sha256'])
def test_source_hash_drift_rejected(field):
    gt,old=sources();gt['rows'][0]['hashes'][field]='d'*64
    with pytest.raises(ValueError):training_sources(gt,old,1)


def test_test_image_cannot_be_registered_as_training():
    gt,old=sources();gt['rows'][0]['source']=old['files'][0]['source']='test_101'
    with pytest.raises(ValueError):training_sources(gt,old,1)


class Helper:
    def capture_one(self,model,restorer,rgb,box,config,seed,operators):
        tensor,restore,grid,crop=operators
        assert restore(restorer,'image',seed)=='restored'
        return {'boundary':'value'},{'seed_seen':seed}


def test_zero_operator_calls_actual_probe_and_records_state():
    def probe(restorer,image,seed,mode,audit):
        assert mode=='zero' and seed==123
        audit.update(noise_nonzero=0,initial_equals_condition=True,q_sample_calls=1)
        return 'restored',None
    arrays,audit=capture_n1_window(Helper(),None,None,None,None,{},123,operators=(None,None,None),zero_probe=probe)
    assert audit['actual_local_zero']['noise_nonzero']==0
    assert arrays['boundary']=='value'


def test_false_zero_initial_state_fails():
    def probe(restorer,image,seed,mode,audit):
        audit.update(noise_nonzero=1,initial_equals_condition=False,q_sample_calls=1)
        return 'restored',None
    with pytest.raises(RuntimeError):
        capture_n1_window(Helper(),None,None,None,None,{},123,operators=(None,None,None),zero_probe=probe)
