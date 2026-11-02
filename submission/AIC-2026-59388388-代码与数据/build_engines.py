# -*- coding: utf-8 -*-
"""将三份完整模型权重构建为五个FP32 TensorRT引擎。"""
from training.engine_export import main


if __name__ == '__main__':
    main()
