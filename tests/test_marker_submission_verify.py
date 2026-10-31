# -*- coding: utf-8 -*-
"""仅虚构100图：提交成员完整性、PNG格式和严格类别映射。"""
from copy import deepcopy
import hashlib
import json
import struct
import zipfile

import cv2
import numpy as np
import pytest

from tools.verify_marker_submission import report_rows, validate_prediction, verify_archive


def fixture(tmp_path):
    ids = np.array([[0,1,1],[65535,65535,0]], np.uint16)
    success, encoded = cv2.imencode('.png', ids); assert success
    png = encoded.tobytes(); classes = b'{"1":0,"65535":1}'
    hashes = dict(png_sha256=hashlib.sha256(png).hexdigest(),
                  classes_sha256=hashlib.sha256(classes).hexdigest())
    report = dict(format='marker_reflect_run_v1', complete=True, full100_complete=True,
        all_baselines_exact=True, model_unchanged=True, restorer_unchanged=True,
        no_optimizer_steps=True, no_seams=True, reflection_padding=32, training=False,
        best_deployment_changed=False, parameter_count=1000, complete_source_count=100,
        selected_full_indices=list(range(100)), images=[])
    for i in range(100):
        report['images'].append(dict(source='synthetic_%03d'%i, full_source_index=i,
            view='patch1024', shape=[2,3], output=hashes.copy(), baseline_final_ids_exact=True,
            baseline_rawvote_classes_exact=True, baseline_phases_blend_exact=True,
            replay=dict(original_marker_exact=True,original_watershed_exact=True,
                        original_elevation_exact=True,original_rawvote_exact=True)))
    report_path=tmp_path/'report.json'; report_path.write_text(json.dumps(report),encoding='utf8')
    archive_path=tmp_path/'candidate.zip'
    with zipfile.ZipFile(archive_path,'w',zipfile.ZIP_DEFLATED) as archive:
        for row in report['images']:
            archive.writestr(row['source']+'_inst.png',png)
            archive.writestr(row['source']+'_class.json',classes)
    return archive_path,report_path,report,png,classes


def test_full100_byte_and_native_contract(tmp_path):
    archive,path,_,_,_=fixture(tmp_path)
    result=verify_archive(archive,path)
    assert result['passed'] and result['crc_ok'] and result['images']==100 and result['files']==200
    assert result['member_sha256_exact'] and result['instance_class_ids_exact']
    assert all(row['maximum_id']==65535 and row['dtype']=='uint16' for row in result['checked'])
    assert result['cpu_only'] and not result['inference_performed'] and not result['uploaded']


@pytest.mark.parametrize('field',['full100_complete','all_baselines_exact','model_unchanged',
                                 'restorer_unchanged','no_seams'])
def test_incomplete_or_changed_run_rejected(tmp_path,field):
    _,_,report,_,_=fixture(tmp_path); report[field]=False
    with pytest.raises(ValueError,match='unfrozen'): report_rows(report)


def test_duplicate_and_nonflat_zip_members_rejected(tmp_path):
    archive,path,_,png,_=fixture(tmp_path)
    with zipfile.ZipFile(archive,'a') as target:
        with pytest.warns(UserWarning): target.writestr('synthetic_000_inst.png',png)
    with pytest.raises(ValueError,match='unique flat'): verify_archive(archive,path)
    archive,path,_,png,_=fixture(tmp_path)
    with zipfile.ZipFile(archive,'a') as target: target.writestr('nested/extra.png',png)
    with pytest.raises(ValueError,match='unique flat'): verify_archive(archive,path)


def test_member_bytes_must_match_row_receipt(tmp_path):
    archive,path,report,_,_=fixture(tmp_path)
    report['images'][5]['output']['classes_sha256']='0'*64
    path.write_text(json.dumps(report),encoding='utf8')
    with pytest.raises(ValueError,match='member bytes'): verify_archive(archive,path)


def test_archive_crc_failure_is_rejected_before_member_validation(tmp_path):
    archive,path,_,_,_=fixture(tmp_path)
    with zipfile.ZipFile(archive) as source:
        members=[(name,source.read(name)) for name in source.namelist()]
    stored=tmp_path/'stored.zip'
    with zipfile.ZipFile(stored,'w',zipfile.ZIP_STORED) as target:
        for name,data in members: target.writestr(name,data)
    payload=bytearray(stored.read_bytes())
    name_length,extra_length=struct.unpack_from('<HH',payload,26)
    payload[30+name_length+extra_length]^=1
    stored.write_bytes(payload)
    with pytest.raises(ValueError,match='CRC'): verify_archive(stored,path)


@pytest.mark.parametrize('labels',[b'{"1":0,"1":1,"65535":1}',b'{"1":false,"65535":1}',
    b'{"1":0.0,"65535":1}',b'{"1":2,"65535":1}',b'{"01":0,"65535":1}',
    b'{"1":0}',b'{"1":0,"65535":1,"3":0}',b'{"1":NaN,"65535":1}'])
def test_strict_json_and_exact_positive_mapping(tmp_path,labels):
    _,_,_,png,_=fixture(tmp_path)
    with pytest.raises(ValueError): validate_prediction(png,labels,[2,3],'synthetic')


def test_png_ihdr_dtype_and_shape_not_only_decoded_image(tmp_path):
    _,_,_,png,classes=fixture(tmp_path)
    with pytest.raises(ValueError,match='shape'): validate_prediction(png,classes,[3,2],'synthetic')
    good,encoded=cv2.imencode('.png',np.ones((2,3),np.uint8)); assert good
    with pytest.raises(ValueError,match='16-bit'): validate_prediction(encoded.tobytes(),b'{"1":0}',[2,3],'synthetic')
    corrupted=bytearray(png); corrupted[12:16]=b'BAD!'
    with pytest.raises(ValueError,match='IHDR'): validate_prediction(bytes(corrupted),classes,[2,3],'synthetic')


def test_report_cohort_duplicate_and_per_source_gate_rejected(tmp_path):
    _,_,report,_,_=fixture(tmp_path)
    duplicate=deepcopy(report); duplicate['images'][1]['source']=duplicate['images'][0]['source']
    with pytest.raises(ValueError,match='Duplicate source'): report_rows(duplicate)
    wrong=deepcopy(report); wrong['images'][0]['replay']['original_marker_exact']=False
    with pytest.raises(ValueError,match='actual marker'): report_rows(wrong)
