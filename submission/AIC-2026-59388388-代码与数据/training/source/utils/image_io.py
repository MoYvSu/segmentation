# -*- coding: utf-8 -*-
"""保留OpenCV编解码参数，通过Python文件接口支持Windows中文路径。"""
from pathlib import Path

import cv2
import numpy as np


def read_image(path, flags=cv2.IMREAD_COLOR):
    path = Path(path)
    content = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), flags)
    if image is None:
        raise ValueError(f'Could not decode image: {path}')
    return image


def write_image(path, image):
    path = Path(path)
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise RuntimeError(f'Could not encode image: {path}')
    path.write_bytes(encoded.tobytes())
