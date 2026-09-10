# 掩码投影梯度修复后的同条件重跑

用户已同意先重跑当前监督基线，训练结束后再决定是否加入一致性自监督。
首轮结果与缺陷证据见 [结果分析](MASK_SET_RESULTS_20260910.md)。

本轮已完成120轮，按验证损失选出的联训最优为e103。最终输出比首轮改善，但仍低于主线；
训练与纯内容域实例稳定性诊断见[修复版结果及一致性学习决策](MASK_SET_AMPFIX_RESULTS_20260910.md)。

本轮北京时间2026-09-10 **19:21:15**启动，PID `24717`。独立服务器项目：
`/root/autodl-tmp/segmentationv2_maskset_ampfix_20260910`。
输出：`outputs/20260910_192115_masksetfix`。

## 唯一训练变化

从旧run的136文件源码快照建立新目录，只替换`models/mask_set.py`与对应回归测试。
运行逻辑只改初始掩码投影的梯度模式，避免BF16缓存意外冻结投影；注意力阈值仍detach。
修复源码SHA256：`820f710a9d76b1ba7362f9d2190f3cd87f84d8ec1bcba900cd6d02288b6af721`。

配置逐键核对，除项目/输出目录外与上一轮完全一致：原新GT、26/6划分、seed42、1024输入、
512掩码网格、256query、BF16、每轮64增强样本、batch1；预热60轮，恢复本轮验证损失最优，
再联训LoRA60轮。阶段按确定性验证损失选优，不按六图最终代理分数选权重。

复用SSL LoRA SHA256：`e5b837fce1707a682d4a149645dd3a210556f737b8f5778d9d4c8b275a34cb49`。
decoder从相同随机种子重新初始化，不读取旧e118或两步smoke的权重。没有重训SSL，没有
引入一致性/伪标签/额外无标签训练流；推理设置也保持不变。

旧项目、源码与权重保留，数据、SAM2基座/源码和SSL资产仅通过新目录内的链接复用。

## 启动与产物

实际命令（已启动，勿重复）：

```bash
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
SEGMENTATION_PROJECT_ROOT=/root/autodl-tmp/segmentationv2_maskset_ampfix_20260910 \
/root/miniconda3/envs/sam2_env/bin/python -u train_mask_set.py \
  --config config/train/mask_set_clean60.yaml \
  --output-dir outputs/20260910_192115_masksetfix
```

run内保存`launch.json`、`preparation.json`、`source_manifest.json`、`source_snapshot.tar.gz`及
启动脚本，训练自身保存解析配置、split、SSL/GT来源、metrics.csv、run.log、best_warmup.pth、
best_joint.pth、last_mask_set.pth及每10轮权重。正常结束写`COMPLETED.json`，异常写`FAILED.json`。
本地启动资产位于[output/20260910_mask_set_rerun](../output/20260910_mask_set_rerun)。

启动前沿用同一修复源码已通过的14项CPU测试、GPU BF16同权重推理一致性和真实图像两阶段
单步验证，不重复已完成的长拟合。正式运行启动后只核验一次投影参数更新，不设置跟踪任务。
该次核验读取到预热e5：全部148个decoder参数均有优化器状态；原先冻结的6个掩码投影参数
各完成320步更新，相比旧冻结初值的最大绝对变化为0.0074–0.0229。预热LoRA与原SSL保持一致，
进程正常、无FAILED标志。证据为run内及本地启动目录的`startup_verification.json`。
e5验证损失7.74399仅作运行记录，不用于宣布最终效果改善。
用户通知训练完成后，再对本轮best_joint做同六图、新GT、原尺寸最终输出比较，并据此决定
下一步是否加入实例一致性学习。

评估入口（训练完成后执行）：

```bash
/root/miniconda3/envs/sam2_env/bin/python tools/evaluate_mask_set.py \
  --config config/train/mask_set_clean60.yaml \
  --checkpoint outputs/20260910_192115_masksetfix/best_joint.pth \
  --output-dir outputs/20260910_192115_masksetfix_evaluation --device cuda
```

最终六图匹配实例608、匹配IoU 0.7543、GT惩罚IoU 0.4868、本地代理总分69.48。
建议下一轮做有筛选的实例一致性对照实验，目前尚未启动；E10a+G4b继续作为部署主线。
