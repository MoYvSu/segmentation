# E7b 语义专项训练（2026-08-27）

## 动机

E7a 在固定 G4b 几何的 6178 个测试实例中仅改变 207 个类别（3.35%）。adaptive-core
带来极小的匹配正信号，但 ferrite 面积代理恶化；Lab 后处理还产生明显的 ferrite 单向偏置。
因此后处理投票只保留为审计工具，下一步改为训练语义表征。

## E7b-A 固定项

- 初始化：`outputs/stage2_v6/best_model_stage2.pth`；不从零开始；
- 只训练 `seg_fpn + seg_branch`；boundary、center、affinity 和 LoRA 全冻结；
- 部署几何固定为 G4b、`high=0.65`、seal2、受阻分水岭和 hard majority；
- 20 epoch、每 epoch 62 个 labeled step；checkpoint 按验证 semantic mIoU 选择；
- 固定无标签 holdout 只用于 monitor，不参与 checkpoint 选择。

配置：`config/train/stage2_semantic_e7b_decoder20.yaml`。

## 新监督

### 实例等权核心损失

历史 BCE/Dice 按像素平均，大实例会淹没小实例。E7b 从 LabelMe polygon 同步生成实例图，
并在 letterbox、裁剪、翻转和旋转中保持同一几何变换。每个实例先排除 GT 边界附近 3 px，
计算核心 BCE，再按实例等权、按 ferrite/pearlite 两类等权汇总。若细小或狭长实例腐蚀后
少于 12 px，则退回该实例全部像素，避免小实例失去监督。

总语义监督为：

`pixel BCE + 0.30 * symmetric Dice + 0.50 * instance-balanced core BCE`。

### 定向增强

学生有标签输入新增 target-aware dark-rim：只改变边界邻域亮度，不移动标签，随机宽度
2--6 px、暗化 10%--30%。它与 gamma、白平衡、轻度失焦、降采样、低频照明和低概率划痕
随机组合，用于模拟“小实例被粗黑边包围”的已观察失败模式。

### 无标签一致性

EMA 教师只在 `p<=0.15` 或 `p>=0.85` 的高置信像素提供语义一致性，权重 0.05，4 epoch
ramp-up，温度 0.70。边界一致性为零且边界分支冻结，避免语义实验反向改变 G4b 几何。

## 几何兼容契约

旧 affinity checkpoint 的 digest 同时包含语义 decoder 和 LoRA，导致合法的 decoder-only
更新被拒绝。现在允许配置 `semantic_decoder_override_base`：只有新 checkpoint 与 V6 的
LoRA 张量逐字节一致时，才允许替换语义 decoder；任何 LoRA 变化仍会拒绝复用 G4b。

部署配置：`config/experiments/affinity_g4b_high065_semantic_e7b.yaml`。

## 启动与判定

```bash
conda activate sam2_env
python train_stage2.py --config config/train/stage2_semantic_e7b_decoder20.yaml
```

训练后固定几何推理：

```bash
python tools/run_affinity_submission.py \
  --config config/experiments/affinity_g4b_high065_semantic_e7b.yaml
```

晋级需要同时满足：验证 ferrite/pearlite mIoU 不退化、小实例类别错配减少、ferrite 平均面积
代理不恶化，并且无标签 monitor 未出现置信度整体塌向 0.5。测试集仅作无 GT 目检和提交前
稳定性审计。

## E7b-A 实验结果（2026-08-28）

训练已正常完成，验证 semantic mIoU 最佳点为 epoch 18：`0.82336`；训练开始时为
`0.8255` 左右，最终 epoch 20 为 `0.81770`。core BCE 从约 `0.085` 降至 `0.065`，
说明模型确实拟合了新增监督，但没有转化为验证泛化收益。boundary 分支按设计保持不变。

完整部署闭环（预测语义 → G4b affinity → boundary watershed → hard majority）结果如下：

| 配置 | 阈值 | 代理总分 | 有效匹配 mIoU | 铁素体面积相对误差 | 预测实例数 |
|---|---:|---:|---:|---:|---:|
| G4b/V6 基线 | 0.65 | 88.0551 | 0.8514 | 0.0903 | 1166 |
| E7b-A | 0.65 | 78.8379 | 0.8365 | 0.2598 | 1302 |

因此 E7b-A 不晋级为默认语义 checkpoint，也不应继续在其上开放 LoRA。68 张测试图中，
E7b 与 G4b 共保留 6178 个实例，但有 486 个实例发生类别翻转：365 个 pearlite→ferrite，
121 个 ferrite→pearlite；总 ferrite 实例数由 3805 增至 4049。这个单向偏置会降低铁素体
平均面积，并解释了为什么训练曲线下降而竞赛代理分数变差。

当前回退点仍为 G4b/V6；E7b 仅作为失败实验和后续语义改进的诊断对照。下一步应先针对
语义类别校准和小实例黑边误判设计受控实验，每次只改变一个因素，并以完整部署代理指标
而不是 pixel semantic loss 选择 checkpoint。

## 后续 E7b-B

只有 E7b-A 出现明确正信号后，才从其 best checkpoint 开放后层 LoRA，LoRA LR 约为 decoder
的 0.1 倍。LoRA 一旦变化，必须重新训练/校准 affinity geometry，不能直接复用 G4b。
