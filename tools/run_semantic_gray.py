# -*- coding: utf-8 -*-
"""灰度先验20轮配对实验；短测后顺序训练，自动保存过程图和最终对照。"""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.run_semantic_consistency import run_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/train/semantic_gray.yaml')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run_pipeline(args.config, args.smoke)


if __name__ == '__main__':
    main()
