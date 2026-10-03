# -*- coding: utf-8 -*-
"""冻结语义尺度诊断的原生裁窗与连续 logit 拼接，不生成新实例。"""
from __future__ import annotations

import numpy as np


def tile_starts(length, size, overlap=.25):
    """最后一窗向内平移，图像小于视野时保留真实尺寸，交给 letterbox。"""
    if type(length) is not int or length <= 0 or type(size) is not int or size <= 0:
        raise ValueError('Positive integer length and native window required')
    if not 0 <= overlap < 1:
        raise ValueError('Overlap must be in [0,1)')
    if length <= size:
        return [0]
    step = max(1, int(round(size * (1 - overlap))))
    values = list(range(0, length - size + 1, step))
    if values[-1] != length - size:
        values.append(length - size)
    return values


def tile_boxes(shape, size, overlap=.25):
    """固定逐行顺序，返回 (y0,y1,x0,x1)，不在整图缩放后裁块。"""
    if len(shape) != 2:
        raise ValueError('Native H,W required')
    h, w = (int(x) for x in shape)
    return [(y, min(y+size,h), x, min(x+size,w))
            for y in tile_starts(h,size,overlap) for x in tile_starts(w,size,overlap)]


def positive_tent(shape, minimum=.001):
    """像素中心上的正三角权重；外缘也非零，避免拼接留下无监督缝隙。"""
    if len(shape) != 2 or any(int(v) <= 0 for v in shape) or not 0 < minimum <= 1:
        raise ValueError('Positive H,W and weight floor in (0,1] required')
    axes = []
    for length in shape:
        length = int(length)
        position = (np.arange(length,dtype=np.float32)+np.float32(.5))/np.float32(length)
        axes.append(np.maximum(np.float32(minimum),1-np.abs(position*2-1)))
    return np.ascontiguousarray(axes[0][:,None]*axes[1][None,:],dtype=np.float32)


class NativeLogitBlend:
    """原坐标 float32 logit 加权平均；先拼 logit，之后统一 sigmoid。"""
    def __init__(self, shape, minimum_weight=.001):
        if len(shape) != 2 or any(int(v) <= 0 for v in shape):
            raise ValueError('Positive native shape required')
        if not 0 < minimum_weight <= 1:
            raise ValueError('Weight floor must be in (0,1]')
        self.shape = tuple(int(x) for x in shape)
        self.minimum_weight = float(minimum_weight)
        self.numerator = np.zeros(self.shape,dtype=np.float32)
        self.denominator = np.zeros(self.shape,dtype=np.float32)
        self.boxes = []

    def add(self, box, logits):
        if len(box) != 4 or any(type(v) is not int for v in box):
            raise ValueError('Integer y0,y1,x0,x1 required')
        y0,y1,x0,x1 = box
        if not (0 <= y0 < y1 <= self.shape[0] and 0 <= x0 < x1 <= self.shape[1]):
            raise ValueError('Tile exceeds native output grid')
        if tuple(box) in self.boxes:
            raise ValueError('Repeated tile would alter the declared weights')
        logits = np.asarray(logits)
        if logits.shape != (y1-y0,x1-x0) or logits.dtype != np.float32 or not np.isfinite(logits).all():
            raise ValueError('Finite native-tile float32 logits required')
        weights = positive_tent(logits.shape,self.minimum_weight)
        self.numerator[y0:y1,x0:x1] += logits*weights
        self.denominator[y0:y1,x0:x1] += weights
        self.boxes.append(tuple(box))

    def finish(self):
        if not self.boxes or not np.all(self.denominator > 0):
            raise ValueError('Every native pixel must receive strictly positive coverage')
        result = self.numerator/self.denominator
        if not np.isfinite(result).all():
            raise ValueError('Blended logits must be finite')
        return np.ascontiguousarray(result,dtype=np.float32)

    def receipt(self):
        return dict(tiles=len(self.boxes),native_shape=list(self.shape),
                    minimum_denominator=float(self.denominator.min()),
                    maximum_denominator=float(self.denominator.max()),
                    uncovered_pixels=int((self.denominator<=0).sum()),
                    arithmetic='CPU float32 positive-tent weighted native logits; sigmoid after stitching',
                    minimum_axis_weight=self.minimum_weight)


def prepare_native_tile(rgb, box, network_size, device):
    """与实际全图 prepare_image 相同的 RGB/255 和 reflect letterbox。"""
    import torch
    from data.dataset import letterbox
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError('Source must be native uint8 RGB')
    y0,y1,x0,x1 = box
    if not (0 <= y0 < y1 <= rgb.shape[0] and 0 <= x0 < x1 <= rgb.shape[1]):
        raise ValueError('Native RGB box out of bounds')
    native = rgb[y0:y1,x0:x1]
    image,scale,ph,pw = letterbox(native,network_size)
    tensor = torch.from_numpy(image).permute(2,0,1).float().unsqueeze(0).to(device)/255.
    return tensor,int(ph),int(pw),dict(box=list(box),native_shape=list(native.shape[:2]),
        network_shape=[network_size,network_size],resize_scale=float(scale),
        pad_h=int(ph),pad_w=int(pw),padding='bottom/right BORDER_REFLECT',
        input_stride=list(tensor.stride()))
