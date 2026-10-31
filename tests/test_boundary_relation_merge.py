# -*- coding: utf-8 -*-
"""关系评分单轮合并：合成原生图，无网络/GT/WS。"""
import json
from copy import deepcopy
import numpy as np
import pytest

from utils.boundary_relation_merge import merge_relation_pairs


def candidate(a,b,score=.01):return dict(pair_ids=[a,b],keep_score=score)


def stripes(gap=1):
    ids=np.zeros((9,13),np.uint16);ids[1:8,1:6]=9;ids[1:8,6+gap:12]=65535
    return ids,{'9':1,'65535':1}


@pytest.mark.parametrize('threshold,candidates',[(0,[candidate(9,65535)]),(.1,[])])
def test_exact_noop_independent_output_no_compaction(threshold,candidates):
    ids,classes=stripes();old=ids.copy();table=classes.copy()
    out,result,audit=merge_relation_pairs(ids,classes,candidates,threshold=threshold)
    assert np.array_equal(out,ids)and result==classes and not np.shares_memory(out,ids)
    assert out.dtype==np.uint16 and out.max()==65535 and audit['selected_count']==0
    assert np.array_equal(ids,old)and classes==table
    json.dumps(audit,allow_nan=False)


def test_safe_original_onezero_fill_retains_smaller_ID_and_class():
    ids,classes=stripes();old=ids.copy();candidates=[candidate(65535,9)];before=deepcopy(candidates)
    out,result,audit=merge_relation_pairs(ids,classes,candidates)
    assert result=={'9':1}and np.all(out[ids==65535]==9)and np.all(out[1:8,6]==9)
    assert np.array_equal(ids,old)and candidates==before and classes=={'9':1,'65535':1}
    assert audit['filled_seam_pixels']==7 and audit['selected'][0]['merged_area']==old.astype(bool).sum()+7
    assert audit['selected'][0]['components_before']=={'9':{'4':1,'8':1},'65535':{'4':1,'8':1}}
    assert audit['selected'][0]['components_after']=={'4':1,'8':1}
    assert np.all(out[0]==0)and np.all(out[-1]==0)


def test_direct_contact_and_small_big_have_no_area_filter():
    ids=np.ones((6,7),np.uint16)*4;ids[3,3]=2
    out,classes,audit=merge_relation_pairs(ids,{'2':0,'4':0},[candidate(4,2)])
    assert np.all(out==2)and classes=={'2':0}and audit['filled_seam_pixels']==0
    assert audit['selected'][0]['original_areas']==[1,41]


def test_third_ID_junction_zero_and_other_background_remain():
    ids,classes=stripes();ids[0,6]=3;classes['3']=0
    out,result,audit=merge_relation_pairs(ids,classes,[candidate(9,65535)])
    assert out[1,6]==0 and np.all(out[2:8,6]==9)and out[0,6]==3
    assert result=={'9':1,'3':0}and audit['filled_seam_pixels']==6
    assert np.array_equal(out[:,0],ids[:,0])and np.array_equal(out[:,-1],ids[:,-1])


def test_only_third_party_zero_contact_rejects_without_consuming_quota():
    ids=np.zeros((5,7),np.uint16);ids[2,1]=1;ids[2,3]=2;ids[1,2]=3
    ids[3,1]=4  # 1与4有独立直接接触；被拒绝的1/2不可占用1。
    out,classes,audit=merge_relation_pairs(ids,{'1':1,'2':1,'3':1,'4':1},[candidate(1,2,.001),candidate(1,4,.02)])
    assert audit['rejection_reasons']=={'no_safe_binary_contact':1}
    assert audit['selected'][0]['pair_ids']==[1,4]and classes=={'1':1,'2':1,'3':1}
    assert out[2,2]==0 and out[2,3]==2


def test_two_pixel_wide_zero_gap_cannot_be_bridged():
    ids,classes=stripes(2);out,result,audit=merge_relation_pairs(ids,classes,[candidate(9,65535)])
    assert np.array_equal(out,ids)and result==classes
    assert audit['rejection_reasons']=={'not_original_adjacent':1}


@pytest.mark.parametrize('rotation',[0,1])
def test_outer_frame_zero_unchanged_even_when_pair_has_legal_interior_merge(rotation):
    # 上、下外框的0都恰邻两ID，不能因内部已有合法合并而顺手填成前景。
    ids=np.tile(np.array([1,1,0,2,2],np.uint16),(5,1))
    out,classes,audit=merge_relation_pairs(np.rot90(ids,rotation),{'1':1,'2':1},[candidate(1,2)])
    out=np.rot90(out,-rotation)
    assert classes=={'1':1}and audit['selected_count']==1
    assert out[0,2]==0 and out[-1,2]==0 and np.all(out[1:-1,2]==1)
    assert audit['filled_seam_pixels']==3 and audit['original_outer_frame_zero_pixels_unchanged']


def test_global_greedy_selection_has_no_chain_and_is_order_independent():
    ids=np.tile(np.array([1,1,0,2,2,0,3,3,0,4,4],np.uint16),(5,1))
    classes={str(i):1 for i in (1,2,3,4)}
    pairs=[candidate(3,4,.03),candidate(2,3,.01),candidate(1,2,.02)]
    out,table,audit=merge_relation_pairs(ids,classes,pairs)
    other,other_table,other_audit=merge_relation_pairs(ids,classes,list(reversed(pairs)))
    assert np.array_equal(out,other)and table==other_table=={'1':1,'2':1,'4':1}
    assert [r['pair_ids']for r in audit['selected']]==[[2,3]]
    assert [r['pair_ids']for r in other_audit['selected']]==[[2,3]]
    assert np.all(out[:,2]==0)and np.all(out[:,8]==0)and np.all(out[1:-1,5]==2)
    assert out[0,5]==0 and out[-1,5]==0


def test_tie_break_is_canonical_ID_and_repeated_pairs_all_rejected():
    ids=np.tile(np.array([1,1,0,2,2,0,3,3],np.uint16),(5,1));classes={str(i):0 for i in (1,2,3)}
    _,_,audit=merge_relation_pairs(ids,classes,[candidate(2,3),candidate(2,1)])
    assert audit['selected'][0]['pair_ids']==[1,2]
    out,_,audit=merge_relation_pairs(ids,classes,[candidate(2,3),candidate(1,2,.01),candidate(2,1,.001)])
    assert audit['selected'][0]['pair_ids']==[2,3]and audit['rejection_reasons']['duplicate_pair_all_occurrences_rejected']==2
    assert np.all(out[:,2]==0)


def test_preexisting_multi_component_ID_preserved_and_reported():
    ids,classes=stripes();ids[0,0]=9
    out,_,audit=merge_relation_pairs(ids,classes,[candidate(9,65535)])
    assert out[0,0]==9 and audit['selected'][0]['components_before']['9']['4']==2
    assert audit['selected'][0]['components_after']['4']==2


def test_score_is_strict_and_cross_class_or_bad_candidates_skip_safely():
    ids,classes=stripes()
    out,_,audit=merge_relation_pairs(ids,classes,[candidate(9,65535,.1)])
    assert np.array_equal(out,ids)and audit['rejection_reasons']=={'score_not_below_threshold':1}
    out,_,audit=merge_relation_pairs(ids,{'9':0,'65535':1},[candidate(9,65535)])
    assert np.array_equal(out,ids)and audit['rejection_reasons']=={'different_predicted_class':1}
    for score in [float('nan'),float('inf'),-.1,1.1,True,'0.01',None]:
        out,_,audit=merge_relation_pairs(ids,classes,[candidate(9,65535,score)])
        assert np.array_equal(out,ids)and audit['rejection_reasons']=={'invalid_keep_score':1}
        json.dumps(audit,allow_nan=False)
    _,_,audit=merge_relation_pairs(ids,classes,[candidate(9,123)])
    assert audit['rejection_reasons']=={'unknown_instance_id':1}


@pytest.mark.parametrize('bad',[{}, {'9':1}, {'9':1,'65535':True}, {'09':1,'65535':1}, {9:1,65535:1}])
def test_wrong_class_contract_fails_even_disabled(bad):
    with pytest.raises(ValueError):merge_relation_pairs(stripes()[0],bad,[],threshold=0)


@pytest.mark.parametrize('bad',[np.zeros((2,2),np.int32),np.zeros((2,2,1),np.uint16),np.zeros((0,2),np.uint16),[[1,2]]])
def test_bad_instance_type_fails_closed(bad):
    with pytest.raises(ValueError):merge_relation_pairs(bad,{},[])


def test_empty_background_and_readonly_input_are_valid():
    source=np.zeros((1,1),np.uint16);source.flags.writeable=False
    out,classes,audit=merge_relation_pairs(source,{},[])
    assert np.array_equal(out,source)and classes=={}and audit['final_contract_valid']
    ids,classes=stripes();ids.flags.writeable=False
    out,_,_=merge_relation_pairs(ids,classes,[candidate(9,65535)])
    assert out.dtype==np.uint16 and np.unique(out[out>0]).tolist()==[9]


def test_two_independent_pairs_execute_once_preserving_cross_class_zero_seam():
    ids=np.tile(np.array([1,1,0,2,2,0,3,3,0,4,4],np.uint16),(5,1))
    classes={'1':0,'2':0,'3':1,'4':1}
    out,result,audit=merge_relation_pairs(ids,classes,[candidate(2,3,.001),candidate(3,4,.02),candidate(1,2,.01)])
    assert [r['pair_ids']for r in audit['selected']]==[[1,2],[3,4]]and result=={'1':0,'3':1}
    assert np.all(out[1:-1,2]==1)and np.all(out[1:-1,8]==3)and np.all(out[:,5]==0)
    assert np.all(out[0,[2,8]]==0)and np.all(out[-1,[2,8]]==0)
    assert audit['filled_seam_pixels']==6 and audit['rejection_reasons']=={'different_predicted_class':1}
    assert audit['after_class_summary']=={'0':{'instances':1,'pixels':23,'mean_area':23.},
                                         '1':{'instances':1,'pixels':23,'mean_area':23.}}


def test_random_adjacency_matches_existing_candidate_contract():
    # 与既有候选生成器独立实现对照，覆盖原0、三路交点和uint16最大ID。
    from utils.boundary_relation_merge import _geometry
    from utils.boundary_relation_supervision import adjacent_region_pairs
    rng=np.random.default_rng(20261030)
    for shape in [(1,1),(1,7),(7,1),(5,7)]:
        for _ in range(8):
            ids=rng.choice(np.array([0,1,2,5,65535],np.uint16),size=shape)
            pairs,_=adjacent_region_pairs(ids)
            expected={tuple(p['pair_ids'])for p in pairs}
            adjacent,_,seams=_geometry(ids)
            assert {(key//65536,key%65536)for key in adjacent}==expected
            # 安全零缝逐点核验文字合同，不以候选生成器的三路接触作为可填区。
            for key,flat in seams.items():
                for index in flat:
                    y,x=divmod(int(index),shape[1]);neighbor=set()
                    for dy,dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                        yy,xx=y+dy,x+dx
                        if 0<=yy<shape[0]and 0<=xx<shape[1]and ids[yy,xx]>0:neighbor.add(int(ids[yy,xx]))
                    assert ids[y,x]==0 and neighbor=={key//65536,key%65536}
