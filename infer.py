# -*- coding: utf-8 -*-
"""统一推理入口：默认旧C；新训练权重须training-manifest，TensorRT须engine-manifest。"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False,
        epilog='其余参数原样交给所选后端。使用 --backend tensorrt --help 查看引擎参数；'
               '模型及配置默认值保持原入口设置。')
    parser.add_argument('--backend', choices=('torch', 'tensorrt'), default='torch',
                        help='执行后端，默认torch；tensorrt需提供--engine-manifest')
    parser.add_argument('--training-manifest', help='显式使用训练run的verify_export/manifest.json；目前仅支持torch')
    parser.add_argument('-h', '--help', action='store_true', help='显示本入口及所选后端的参数')
    route, arguments = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if route.training_manifest is not None and route.backend != 'torch':
        parser.error('--training-manifest目前只支持torch，与tensorrt/engine-manifest不兼容')
    if route.help:
        parser.print_help()
        print('\n所选后端参数：', flush=True)
        arguments.append('--help')
    if route.training_manifest is not None:
        from tools.infer_trained import main as trained_main
        return trained_main([*arguments, '--training-manifest', route.training_manifest])
    if route.backend == 'tensorrt':
        from tools.infer_accelerated import main as accelerated_main
        return accelerated_main(arguments)

    # compact没有main函数；仅封装其既有parse_args/run及原完成JSON，不复制推理。
    from tools import infer_compact
    result = infer_compact.run(infer_compact.parse_args(arguments))
    print(json.dumps(dict(complete=bool(result.get('complete')), format=infer_compact.FORMAT,
        geometry_mode=result['geometry_mode'], images=len(result['images']),
        actual_compact_counts=result['actual_compact_counts'])), flush=True)
    return result


if __name__ == '__main__':
    main()
