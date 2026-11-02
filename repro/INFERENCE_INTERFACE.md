# 当前推理接口与历史计划

状态更新：2026-11-01。顶层`infer.py`只分流到既有推理入口，`--backend torch|tensorrt`默认`torch`。
训练另用`train_pipeline.py`，不会自动启动推理或将新训练权重晋级为部署模型。

本轮训练导出可直接使用显式入口：

```bash
python infer.py --backend torch --training-manifest outputs/train_run_01/verify_export/manifest.json --input-dir data/test/semifinal --output-dir outputs/infer_trained_01 --dataset semifinal_150
```

该分流调用`tools/infer_trained.py`，按训练清单严格核三权重及NAF合同，记录独立模型身份；
不放宽旧C缺省身份检查，不复制模型或要求另外传模型目录。当前仅支持Torch，不能搭配旧TensorRT引擎。
这是本地训练结果的推理桥，完整提交包格式另定。具体输入与输出约束见[统一指南](../docs/TRAINING_INFERENCE_GUIDE.md)。

## 当前原C与加速候选

| 入口 | 执行内容 | 状态 |
|---|---|---|
| `infer.py`或`infer.py --backend torch` | `tools/infer_compact.py`，默认原C | 旧三权重的等价后处理精简；固定两路全图语义、局部1024 N1/NAF/S0及final200 |
| `tools/infer_fast_readout.py` | 仅增加CPU整数finish快路径 | 三源真实GPU终态与原C逐文件一致；不自动叠加网络编译 |
| `infer.py --backend tensorrt` | `tools/infer_accelerated.py`，全五TensorRT引擎＋CPU整数finish | 150图完成并下载，本地全成员与差异统计核验通过；待本包独立回分 |

固定原backend文件SHA `13b3ab1b…8161f04`、D5a文件SHA `9315da87…17007b`、
旧NAF G state SHA `20a59c39…158e8`；完整身份见[训练参数表](../docs/TRAINING_RECIPE_20261101.md)。
教师32源新三权重虽完成训练与150图推理，但用户回分约80.4，已淘汰；不通过这两个原C入口自动替换。

在项目根目录、GPU推理环境中使用新的私有输出目录。例如Torch原C：

```bash
python infer.py --backend torch \
  --config config/inference/area_floor_200.yaml --dataset semifinal_150 \
  --input-dir <150图目录> --model-dir <旧C的backend及restorer目录> \
  --naf-checkpoint <旧20a59对应checkpoint> \
  --output-dir outputs/inference/C_torch --global-noise zero --geometry-mode C
```

全五引擎候选显式选择TensorRT并提供清单：

```bash
python infer.py --backend tensorrt \
  --config config/inference/area_floor_200.yaml --dataset semifinal_150 \
  --input-dir <150图目录> --model-dir <同一旧C模型目录> \
  --naf-checkpoint <同一旧NAF checkpoint> \
  --engine-manifest <五引擎清单.json> \
  --output-dir outputs/inference/C_TRT_candidate --global-noise zero --geometry-mode C
```

`--help`显示选定后端的实际参数。TensorRT入口不接受F0，要求`C_TensorRT_acceleration_v2`清单完整包含
trunk、semantic、affinity、naf、d5a五角色，相对engine路径、文件SHA及来源模块state SHA逐项匹配；
清单声明FP32、TF32/FP16/INT8全关，来源模块由原严格loader读取后核验，不提供缺失引擎或错误来源的静默回退。
原NN模块对象及state保持，运行结束或异常后恢复临时方法；NAF只替换logits计算体，原Torch sigmoid保留。
这不保证TensorRT与Torch输出逐值相同；本候选仍需独立最终输出与官方分数验收。

当前编码器、语义、affinity和D5a引擎固定B1、1024×1024；只有NAF的256×256窗口支持动态batch1–16。
150图按图依次推理，不是网络batch150。推理入口只消费已构建engine，不会自动构建；
独立公开构建入口为`tools/build_inference_engines.py build`，命令见[统一使用指南](../docs/TRAINING_INFERENCE_GUIDE.md)。
该整理入口本轮仅本地CPU验证，仍待GPU实跑；先前实际构建与适配记录见[全模块加速报告](../docs/FULL_ACCELERATION_20261101.md)。
可选环境依赖见`requirements-tensorrt.txt`。

结果为`results/`下原尺寸单通道uint16实例PNG和逐图类别JSON，P=0/F=1、正ID与类别键集一致、ID≤65535；
审计另存`audit/`。只结果黑盒默认应仅含150对/300成员，不能把审计、模型或源码混入；
打包与无测试GT差异统计可使用`tools/analyze_prediction_flips.py`，不会再次投票或清理实例。
原图、权重、预测、逐图JSON和提交ZIP均保留在被忽略的私有目录，不进入Git。

## 历史计划：E10a/G4b

以下是旧`repro/train.py`编写时固定的历史输入及计划，不是当前原C命令，也不是已实现的`repro/inference.py`：

```text
checkpoint:
  outputs/stage2_semantic_e10a_cold20/best_model_stage2.pth
affinity_checkpoint:
  outputs/affinity_geometry_g4b_gap_weight020/latest_affinity.pth
config:
  config/experiments/affinity_g4b_high065_semantic_e10a_cold.yaml
input_dir: 待推理图像目录
output_dir: 推理复现输出目录
```

当时计划的入口为（该文件未实现，不能直接执行）：

```bash
python repro/inference.py \
  --config config/experiments/affinity_g4b_high065_semantic_e10a_cold.yaml \
  --semantic-checkpoint outputs/stage2_semantic_e10a_cold20/best_model_stage2.pth \
  --affinity-checkpoint outputs/affinity_geometry_g4b_gap_weight020/latest_affinity.pth \
  --input-dir <images> \
  --output-dir <result>
```

历史计划没有创建`repro/inference.py`；E10a/G4b实际路径需查当时的`tools/run_affinity_submission.py`及配置。
当前实际入口统一见本页上半部分，不另维护同名旧包装。原实例PNG、类别JSON、配置快照及uint16/ID限制的合同继续保留。
