# 跳过 G1：配对训练启动记录

2026-09-12 用户要求利用空闲服务器，提前启动 G1 消融训练，不等待 G1 直接部署候选 D 的黑盒成绩。
本轮改变实验排程，训练配方和最终删留规则沿用已审阅方案；D 的成绩仍单独记录，不能将尚未回分当作失败。

## 两组训练

| 组别 | G2 初始化 | 配置 | 输出子目录 |
| --- | --- | --- | --- |
| C 保留 G1 | G1 best，epoch18 | `config/train/affinity_geometry_g2_loss_control.yaml` | `g2_loss_control` |
| X 跳过 G1 | G0-long latest，step1600 | `config/train/affinity_geometry_g2_skip_g1.yaml` | `g2_skip_g1` |

两组均为 30 epoch、batch2、seed42、每轮抽取52个样本，即每组780次计划更新。
人工实例与历史 SAM2 无类别几何样本各占采样概率50%；沿用原始人工 GT、原有增强和 G2 损失，unknown/covered-uncovered ignore。
固定 V6 encoder/LoRA 与八通道 affinity 架构，只训练 geometry decoder；运行配置仅初始化与输出目录不同。
保留 G0/G0-long 暖启动；本实验不能解释为随机初始化即可省去所有人工几何训练。

## 启动位置

- 本地分支：`codex/g1-ablation-loss-20260912`，从 `2dbc9bf` 建立。
- 服务器独立 checkout：`/root/autodl-tmp/segmentationv2_g1_ablation_20260912`，从服务器 `f710a9be6a4e2556eacb77b462a3523601ab25fb` 建立同名分支。
- 环境：`/root/miniconda3/envs/sam2_env/bin/python`。
- 数据、基础权重、SAM2库与 outputs 通过链接复用；原工作目录和原部署权重未改动。
- 统一输出父目录：`outputs/20260912_g1_g2_ablation/`，实际位于原服务器工作目录的 outputs 内。
- 2026-09-12 20:48:15（北京时间）发起串行队列，队列 PID `11912`，见 `training_launch.json`。

队列先运行 C，成功结束后自动运行 X，任一进程非零退出即停止。队列不包含官方提交或训练后评估。
`training_status.json` 记录当前阶段；队列日志为 `training_pair.log`，分组日志为 `g2_loss_control.log` / `g2_skip_g1.log`。
仅确认启动正常，按用户偏好不持续追踪长训练，结束后由用户通知再做固定部署评估。

## 损失选优实现

`affinity_geometry_g1.selection_metric: val_affinity_loss` 为新模式；未指定时保持旧部署/Oracle最大值选优行为。
验证固定人工六图 `172、176、349、623、873、888`；显式核对实际split集合，避免数据目录变化后静默换集合。
使用同网格的原始 FP32 logits 和 torch affinity targets，仅监督两端已覆盖连接。逐图计算后取均值，与训练batch聚合不同。
参数直接读取解析后的 G2 loss：负边权重1.5、困难负边权重1.0、gamma2.0、归一化开启；没有添加新训练损失。

epoch0 同口径计算并作为起点，严格更低才更新 best；相同损失保留较早轮次。
新增空监督、缺通道监督与非有限值检查；固定验证集的逐图和逐通道有效边数保存在 `loss_baseline.json`。
`metrics.csv` 保留逐轮 `val_affinity_loss`，结束后 `selection_summary.json` 给出包括epoch0的完整损失曲线和最优/次优差距。
checkpoint v4 保留实际 `epoch`，另记 `best_selection_epoch`、`best_selection_direction`、`validation_protocol_version`；geometry权重结构不变，严格加载。

逐轮部署验证显式关闭；只保留 best 和最终latest(epoch30)。`monitor_interval=30`、无标签monitor数量0，保留epoch0/30少量人工monitor。
现有checkpoint不含完整优化器和随机状态，不能承诺中断后精确续训。

## 开训前证据

本地相关测试共38项通过：先完成36项，再单独通过2项checkpoint元数据/严格回载检查。未重复全量测试。
服务器完成四个旧权重×同六图的24次前向预检，配置对称、验证集身份、冻结参数、checkpoint严格加载、损失有限性和各通道监督计数均通过。

| 权重 | 统一验证 loss |
| --- | ---: |
| G0-long latest / step1600 | 0.46345190 |
| G1 best / epoch18 | 0.21422542 |
| 历史 G2 best / epoch21 | 0.19600878 |
| 历史 G2 latest / epoch30 | 0.21766641 |

四个权重全部采用同一 C 配置的损失参数；这些数只作当前损失目标的诊断，不是官方成绩。
完整解析配置、实际源码SHA256、旧权重SHA、六图逐图损失和数据来源记录保存在 `training_preflight.json`，本地也已下载副本。

SAM2数据沿用原64图：当前manifest SHA256 `4ed3e2e1fcfd2fa369dd647fc5ce7c869c5bcb7f472e29398ee889b368e40749` 与归档一致，64个mask与64条唯一源图记录对应，类别均null，与人工验证六图无源图重叠。
旧 `approval.json` 未恢复。本轮按用户在既有64图复用方案之后明确要求启动训练的授权执行；该复用依据与历史审核记录缺口分开记录，没有生成历史审核人/时间或虚构已恢复的approval文件。

## 完成后评估

先核对两组日志、loss-best、实际epoch与best epoch，再用固定 E10a/legacy_none/gated-mean/high0.65/seal2/reconstruction8/probability_mean/受阻分水岭完成最终输出比较。
loss-best 是预先声明的候选，latest(epoch30)只用于同更新次数诊断，不按代理分事后换权重。
X 只有在同官方口径不低于 C 且不低于 H 时，才支持在当前暖启动和训练预算下删除独立 G1；否则保留或记录未定。
C/X若best停在epoch0，应按其实际初始化产物解释，不能算新增G2训练收益。
