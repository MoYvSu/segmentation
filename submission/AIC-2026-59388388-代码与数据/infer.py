# -*- coding: utf-8 -*-
"""纯TensorRT推理入口：输入图片，输出原尺寸实例PNG及珠光体0/铁素体1类别JSON。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time


BASE = Path(__file__).resolve().parent
INCOMPATIBLE_ENGINE = 86


def run_inference(args):
    import torch
    from inference.pipeline import (Pipeline, configure_runtime, engine_files, input_files,
                                     load_engines, read_rgb, save_prediction)
    files = input_files(args.input_dir)
    paths = engine_files(args.engine_dir, args.engine_manifest)
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose a new or empty output directory')
    if not torch.cuda.is_available():
        raise RuntimeError('TensorRT inference requires an NVIDIA CUDA GPU')
    configure_runtime()
    started = time.perf_counter()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.no_grad(), torch.autocast('cuda', enabled=False):
        pipeline = Pipeline(load_engines(paths))
        results = output / 'results'; results.mkdir(parents=True, exist_ok=False)
        for index, path in enumerate(files, start=1):
            then = time.perf_counter()
            rgb = read_rgb(path)
            ids, classes = pipeline.predict(rgb)
            save_prediction(results, path.stem, ids, classes)
            print(f'{index}/{len(files)} {path.stem} {time.perf_counter() - then:.3f}s', flush=True)
        stream.synchronize()
    summary = dict(images=len(files), seconds=time.perf_counter() - started, engine_calls=pipeline.calls)
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def run_with_rebuild(args):
    # 外层仅调度子进程，旧推理进程退出后再构建，释放其全部CUDA资源。
    engine_root = (Path(args.engine_manifest).resolve().parent if args.engine_manifest
                   else Path(args.engine_dir).resolve())
    local = engine_root / 'local'

    def infer(engine_dir, manifest=None):
        command = [sys.executable, '-B', str(BASE / 'infer.py'), '--_worker',
                   '--input-dir', args.input_dir, '--output-dir', args.output_dir,
                   '--engine-dir', str(engine_dir)]
        if manifest:
            command += ['--engine-manifest', manifest]
        return subprocess.run(command).returncode

    code = infer(local) if local.is_dir() else infer(args.engine_dir, args.engine_manifest)
    if code != INCOMPATIBLE_ENGINE:
        return code
    print('TensorRT engine platform mismatch; rebuilding from ' + args.model_dir,
          flush=True)
    with tempfile.TemporaryDirectory(prefix='.engine-rebuild-', dir=engine_root) as work:
        if local.exists():
            local.rename(Path(work) / 'previous')
        subprocess.run([sys.executable, '-B', str(BASE / 'build_engines.py'),
                        '--model-dir', args.model_dir, '--output-dir', str(local)], check=True)
    return infer(local)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True, help='图片目录，也可指定一张图片')
    parser.add_argument('--output-dir', required=True, help='新的输出目录；预测保存在其中的results/')
    parser.add_argument('--engine-dir', default=str(BASE.parent / 'AIC-2026-59388388-模型文件/engines'))
    parser.add_argument('--engine-manifest', help='可选引擎文件清单；仅用于找到五份engine')
    parser.add_argument('--model-dir', default=str(BASE.parent / 'AIC-2026-59388388-模型文件/weights'),
                        help='引擎不兼容时，用于自动重建的三份完整模型权重目录')
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not args._worker:
        return run_with_rebuild(args)
    from inference.engine import EngineCompatibilityError
    try:
        run_inference(args)
    except EngineCompatibilityError as error:
        print(str(error), file=sys.stderr, flush=True)
        return INCOMPATIBLE_ENGINE
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
