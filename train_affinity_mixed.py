# -*- coding: utf-8 -*-
"""单组全图／原生1024混合适配，直接复用原affinity训练循环与损失。"""
import argparse

from data.affinity_mixed import build_datasets
from train_affinity_native import train
from utils.config import load_config


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='config/train/affinity_mixed.yaml')
    p.add_argument('--output', required=True)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args()
    train(load_config(a.config), 1024, a.output, smoke=a.smoke, dataset_builder=build_datasets)
