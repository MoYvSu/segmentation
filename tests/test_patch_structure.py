# -*- coding: utf-8 -*-
import numpy as np
from tools.probe_patch_structure import trusted_ferrite_core,condition_for


def test_trusted_core_excludes_boundary_unknown_and_pearlite():
    ids=np.ones((1024,1024),np.int32);ids[:,512:]=2
    covered=np.ones_like(ids,bool);covered[300:340,100:140]=False
    core=trusted_ferrite_core(ids,covered,{1:1,2:0},8)
    assert core[100,100] and not core[100,300]
    assert not core[100,251] and not core[156,60] and not core[3,100]


def test_identity_does_not_mutate_clear_source():
    target=np.full((1024,1024,3),.6,np.float32);result=condition_for(target,'identity')
    assert np.array_equal(result,target)
    result[0,0]=0
    assert np.all(target==.6)
