# -*- coding: utf-8 -*-
"""图像路径变化不改变uint16掩码、JPEG解码和编码参数。"""
import cv2
import numpy as np
import pytest

from utils.image_io import read_image, write_image


def test_unicode_uint16_png_and_original_encoding_are_identical(tmp_path):
    image = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    ascii_path = tmp_path/'reference.png'
    assert cv2.imwrite(str(ascii_path), image)
    directory = tmp_path/'中文目录'
    directory.mkdir()
    unicode_path = directory/'实例掩码.png'
    write_image(unicode_path, image)
    assert ascii_path.read_bytes() == unicode_path.read_bytes()
    actual = read_image(unicode_path, cv2.IMREAD_UNCHANGED)
    assert actual.dtype == np.uint16 and actual.ndim == 2 and np.array_equal(actual, image)


def test_jpeg_decode_keeps_original_opencv_pixels(tmp_path):
    image = np.random.default_rng(731).integers(0, 256, (41, 67, 3), dtype=np.uint8)
    path = tmp_path/'original.jpg'
    assert cv2.imwrite(str(path), image)
    assert np.array_equal(read_image(path), cv2.imread(str(path), cv2.IMREAD_COLOR))
    chinese = tmp_path/'中文原图.jpg'
    chinese.write_bytes(path.read_bytes())
    assert np.array_equal(read_image(chinese), read_image(path))


def test_missing_and_invalid_files_are_explicit(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_image(tmp_path/'missing.png')
    invalid = tmp_path/'broken.png'
    invalid.write_bytes(b'not an image')
    with pytest.raises(ValueError, match='Could not decode'):
        read_image(invalid)
