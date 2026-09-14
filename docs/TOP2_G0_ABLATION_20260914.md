# X 的 mean/top2 对照与 G0/G0-long 联合消融

## 授权与实验问题

2026-09-14 用户要求先完成 X 的 mean/top2 推理对照，然后启动删除 G0/G0-long 的训练。
两者都是两张图上的过拟合暖启动：G0 为 train_001/002 的 400 步，G0-long 再训练 1600 步。
本次把两段作为一个整体消融；若退步，再考虑只补回 G0。不能从联合消融单独解释每段贡献。

X 已跳过独立 G1，其用户回报黑盒为 mIoU 0.8438、面积项 0.8464、总分 84.510。
新训练保留 V6 边界 FPN 初始化、原人工 GT、历史64张 SAM2 几何数据与原 G2 配方。
本轮不引入新 GT、不重新训练 SSL/LoRA，也不提前把 top2 切为默认部署。

## 推理对照

- 固定 X loss-best epoch16：`outputs/20260912_g1_g2_ablation/g2_skip_g1/best_affinity.pth`。
- SHA256：`5f29f8a26258b95b195ce0a8fece66675982036ef258993c8b98f318692c1579`。
- 固定 E10a epoch19 与 V6 encoder/LoRA、legacy_none、1024输入/512输出。
- 固定 gated、high0.65、low0.45、seal2、局部重建8步、probability_mean 和完整受阻分水岭。
- 唯一推理变量：`affinity_deployment.short_reduction: mean / top2`。
- top2 对前四个短程方向的**边界概率**取最大两项平均；不是对 affinity 取最大两项，也不是修改 fusion_mode。
  在 gated 模式下，它同时改变短程证据与依赖该证据的长程门控，这是同一读出改动的作用。

mean 复用已回分的68图提交与固定六图输出；重跑 test_001/024/044 核对最终文件一致性。
top2 完成同六图诊断和68图正式格式推理，生成平铺 uint16 PNG/类别 JSON 提交 ZIP。
六图：172、176、349、623、873、888，评价使用既有补缝 GT、unknown ignore，仅作内部诊断。
正式测试推理不读取 GT；官方回分仍由用户提交后提供。

配置：`config/experiments/affinity_x_mean_20260914.yaml`、`affinity_x_top2_20260914.yaml`。
实验脚本与少量产物在 `output/20260914_top2_g0_ablation/`；服务器使用 `outputs/` 同名目录。

## 删除两图暖启动：Y

```text
X：V6 boundary FPN → G0 → G0-long → G2
Y：V6 boundary FPN ──────────────→ G2
```

Y 显式使用 `geometry_init_mode: v6_boundary_fpn`，`geometry_init_checkpoint: null`。
完整随机构造 affinity decoder 后仅严格复制 V6 boundary_fpn；上采样层及 affinity 输出层保持种子初始化。
构造顺序与 G0 第一次训练前一致，不读取 G0/G0-long/G1/G2 训练后的 geometry 权重。

固定 seed42、30 epoch、batch2、52次抽取/轮、人工/SAM2各50%采样概率，即780次计划参数更新。
只训练 geometry decoder，V6/LoRA冻结；负边权重1.5、困难负边权重1.0、gamma2.0与原增强均不变。
保留原人工六图验证损失选优、epoch0基线、只保存best及最终latest；不按六图部署代理分挑epoch。
删阶段也减少了总训练量，若失败不能直接证明两图监督在任何同预算配方下不可替代。

配置：`config/train/affinity_geometry_g2_skip_g0.yaml`。
输出：`outputs/20260914_top2_g0_ablation/g2_skip_g0/`。
训练后先按固定 mean 配方与已有X比较；若之后正式采用top2，必须对双方统一应用，避免同时改变训练和推理后归因。

## 环境与状态

- 本地分支：`codex/g0-ablation-top2-20260914`，基于 `946605f`。
- 推理复用已验证服务器checkout：`/root/autodl-tmp/segmentationv2_g1_ablation_20260912`。
- 新训练独立checkout：`/root/autodl-tmp/segmentationv2_g0_ablation_20260914`，基础为 `f710a9b`，只同步必要实现与配置。
- 不整体上传本地旧代码：服务器历史数据读取/语义推理实现比本地checkout新，整目录覆盖会破坏已验证评估入口。
- 数据、基础权重、SAM2库与outputs通过链接复用，不复制数据集。
- 2026-09-14首次检查：RTX4090空闲，数据盘37%，无需本轮清理。
- 复用此前用户已批准的64图SAM2批次，不新增候选；历史approval文件缺口沿用此前记录，不补造审核信息。

## 已完成推理与启动预检

2026-09-14北京时间12:03:38发起推理，12:08:16完成全部推理、打包与CPU分析。
原mean提交ZIP的136个文件与复用预测逐文件字节一致，另重跑三张测试图也逐字节复现。
两组实际manifest除short_reduction外的部署合同一致，并核验了完整解析配置，包含manifest未列出的后处理参数。

| 同六图内部诊断 | mean | top2 |
| --- | ---: | ---: |
| 有效匹配 / 909 | 720 | 733 |
| 有效匹配mIoU | 0.842912 | 0.836772 |
| GT惩罚mIoU | 0.667653 | 0.674757 |
| 铁素体均面积相对误差 | 28.3474% | 30.9320% |
| 整体代理分 | 77.9719 | 76.3726 |
| 逐图平均代理分 | 78.2676 | 76.9475 |

增加的13个有效匹配均为珠光体；铁素体匹配数仍为431，铁素体预测数在已知域由697增至724。
这个结果支持“部分额外切分，同时存在碎分/面积代价”的诊断，不证明黑盒必然变差。
测试68图总实例数由6018增至6134；固定三图中test_001为99→103、test_024为82→82、test_044为68→66。
top2不保证每图都增加实例，不能由实例数直接判断优劣。三图目检整体分区相近，局部分裂与合并均有变化。
继续保留mean作为训练消融的固定推理基准，top2交用户做一次正式黑盒验证。

- 新提交包：[submission_X_best_e16_top2_high065_20260914.zip](../output/20260914_top2_g0_ablation/submission_X_best_e16_top2_high065_20260914.zip)。
- 68图、136个平铺文件、uint16单通道与原尺寸/类别ID对应校验通过；服务器和下载后CRC均通过。
- 大小3,173,405 bytes，SHA256 `7020696e285a48f50348ba423c27b494fae1ce130d4c108e8d577c6f6f1df58d`。
- [三图对比](../output/20260914_top2_g0_ablation/test_mean_top2_comparison.jpg)、[汇总](../output/20260914_top2_g0_ablation/summary.json)。

本地50项相关CPU测试通过，覆盖旧checkpoint严格兼容、新初始化与原G0训练前的参数/RNG一致性、冻结和保存回载。
服务器真实SAM2/V6预检也确认完整geometry参数与原G0首次更新前逐张量一致，CPU随机数状态一致。
只有39个geometry张量可训练，共5,670,024参数；V6所有参数冻结；原验证划分、64图SAM2身份与52次抽样预算均通过。
初始验证loss为0.6929301520，四类监督协议字段及正/负边数与X的旧预检一致；此值只表示随机输出层起点，不是最终质量判断。

训练wrapper PID6770，于12:08:24开始预检，12:08:48执行30轮训练命令。
`training_preflight.json`记录实际代码SHA、配置、初始化来源和数据复用依据，`training_status.json`与`g2_skip_g0.log`记录状态。
启动检查确认12:10:39、12:10:50、12:10:59分别完成前三轮，验证loss为0.68021562、0.65882495、0.63512849；未见异常退出。
早期loss下降只用于确认能正常学习，不作为删除两段暖启动成功的结论。启动前约两分钟主要为数据加载和epoch0检查。
长训练不安排持续追踪或自动选取新推理模式，结束后由用户通知再分析；本记录不表示训练已经完成或top2已有黑盒成绩。
