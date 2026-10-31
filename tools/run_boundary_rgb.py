# -*- coding: utf-8 -*-
"""顺序执行RGB补边预检、固定批可学习性检查和独立短训；不出全量包。"""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_boundary_restore import write_json
from utils.config import load_config


def paired_preflight_check(first, second):
    for item in (first, second):
        if item.get('complete') is not True or item.get('no_parameter_updates') is not True:
            raise RuntimeError('预检回执未完成')
    if first['plan'] != second['plan']:
        raise RuntimeError('RGB/T-only采样计划不相同')
    a, b = first['initialization_identity'], second['initialization_identity']
    if a['dataset']['real_input_sources'] != b['dataset']['real_input_sources']:
        raise RuntimeError('两组RGB/T/新GT源绑定不相同')
    if a['initial_discriminator_sha256'] != b['initial_discriminator_sha256']:
        raise RuntimeError('两组判别器初态不相同')
    ra, rb = json.loads(json.dumps(a['recipe'])), json.loads(json.dumps(b['recipe']))
    for recipe in (ra, rb):
        recipe.pop('input_mode')
        recipe.pop('output_dir')
        recipe['inference'].pop('input_mode')
    if ra != rb:
        raise RuntimeError('两组除输入模式/输出路径外有额外配方差异')
    return dict(complete=True, plan_equal=True, source_binding_equal=True,
                discriminator_initial_equal=True, recipe_only_input_mode_differs=True,
                plan_sha256=first['plan']['plan_sha256'])


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'queue_started.json').exists():
        raise ValueError('此队列已有启动回执；先核对状态，不自动重开或覆盖')
    configs = [Path(p).resolve() for p in args.configs]
    modes = [load_config(p)['boundary_naf']['input_mode'] for p in configs]
    if modes != ['rgb_boundary', 'boundary']:
        raise ValueError('本轮队列必须按RGB候选、单通道控制顺序独立执行')
    began = time.perf_counter()
    write_json(output / 'queue_started.json', dict(pid=os.getpid(), configs=[str(p) for p in configs],
        modes=modes, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        no_full_test_inference=True, no_promotion=True, no_checkpoint_inheritance=True))
    env = dict(os.environ, PYTHONUNBUFFERED='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    completed = []
    try:
        def invoke(config, destination, phase, updates=None):
            command = [sys.executable, '-u', str(ROOT / 'train_boundary_rgb.py'), '--config', str(config),
                       '--output', str(destination), '--mode', phase,
                       '--device', 'cpu' if phase == 'preflight' else args.device]
            if updates is not None:
                command += ['--smoke-updates', str(updates)]
            write_json(output / 'queue_status.json', dict(complete=False, phase=phase,
                config=str(config), command=command, completed=completed))
            print(json.dumps(dict(event='queue_phase', phase=phase, config=str(config)), ensure_ascii=False), flush=True)
            subprocess.run(command, cwd=ROOT, env=env, check=True)

        preflights = []
        for config, name in zip(configs, ('rgb', 'boundary')):
            destination = output / ('preflight_' + name)
            invoke(config, destination, 'preflight')
            preflights.append(json.loads((destination / 'preflight.json').read_text(encoding='utf-8')))
        pair = paired_preflight_check(*preflights)
        write_json(output / 'paired_preflight.json', pair)
        for config, name in zip(configs, ('rgb', 'boundary')):
            destination = output / ('smoke_' + name)
            invoke(config, destination, 'smoke', updates=args.smoke_updates)
            receipt = json.loads((destination / 'smoke.json').read_text(encoding='utf-8'))
            if receipt.get('passed') is not True:
                raise RuntimeError('固定批可学习性未通过：' + name)
        # 正式模型由新进程从公共随机初态重建，不读取任一smoke权重。
        for config, name in zip(configs, ('rgb', 'boundary')):
            destination = output / name
            invoke(config, destination, 'train')
            receipt = json.loads((destination / 'complete.json').read_text(encoding='utf-8'))
            if receipt.get('complete') is not True:
                raise RuntimeError('正式训练未完成：' + name)
            # 训练回执本身已有相对checkpoint；显式覆盖成绝对路径，避免重复关键字报错。
            completed.append({**receipt, 'group': name, 'checkpoint': str(destination / 'latest.pt')})
        result = dict(complete=True, groups=completed, no_full_test_inference=True,
                      elapsed_seconds=time.perf_counter() - began, deployment_replaced=False)
        write_json(output / 'queue_complete.json', result)
        write_json(output / 'queue_status.json', result)
        return result
    except BaseException as error:
        write_json(output / 'queue_failed.json', dict(complete=False, exception=repr(error),
            traceback=traceback.format_exc(), completed=completed, elapsed_seconds=time.perf_counter() - began,
            no_automatic_retry=True, existing_artifacts_preserved=True))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs', nargs=2, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoke-updates', type=int, default=128)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
