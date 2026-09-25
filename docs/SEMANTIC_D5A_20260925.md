# D5a后语义适配与全新简化头

## 问题与方案

[外观响应诊断](D5A_APPEARANCE_20260925.md)表明，D5a输入造成的分类转向主要已出现在
编码特征→粗语义路径，高分辨RGB残差总体抵消一部分偏移。该结果不能证明分类翻转错误，
也不能证明删除残差有益。本轮依用户授权比较两条实际方案：

| 项目 | A / full | B / simple |
|---|---|---|
| 语义结构 | 原FPN、分类层、高分辨RGB残差 | FPN、分类层，无直接RGB修正 |
| 初始化 | S-align e13语义继续适配 | FPN、分类层全新随机初始化 |
| 峰值学习率 | 3e-5 | 1e-4 |
| 共同冻结 | D5a e60、SAM2全部含LoRA、V affinity | 相同 |

B随机初始化为用户明确选择。结构、初始化、学习率同时不同，因此本轮回答哪条实际方案
更适合D5a输入，不能单独归因于删去RGB分支。B仍通过SAM2特征间接获得图像外观，
只是没有额外直接读取RGB的修正分支。

## 数据与训练

- 全32张人工训练源图，全部参与，不划代理或留出集；每轮每图2次，共64更新。
- 新GT为`outputs/experiments/seeded_label_completion_o1/radius_8/derived_maps`，
  manifest为`manual_target_v2`，来源为原人工实例种子的原尺寸8像素范围补缝。
  语义、实例和辅助边缘统一从新GT构造；未知区域、填充区域忽略。实际读取32张，
  5126个有效实例，1024网格已知像素24,299,742，未知及填充9,254,690，未发现对齐错误。
- 无类别SAM2掩码不参与本轮语义监督；affinity冻结，无需重复几何训练。
- 完整图等比letterbox至1024、反射填充；在线退化、翻转和90度旋转，图像与GT同步。
- D5a退化配方期望20%原样、40%均匀模糊、20%旧空间模糊、20%强弱端点过渡；
  复用D5a重采样，另对模糊样本以0.5概率添加sigma 0–0.006的无彩噪声。
  后端dataset显式接入端点sigma场：仅替换配置不会自动调用修复器dataset的独立端点分支。
- 退化图→冻结D5a首步输出→冻结encoder→可训练语义头。D5a和encoder保持FP32，
  语义头训练使用BF16混合精度；部署FP32。
- 每组60轮、3840更新，2轮预热后余弦衰减到峰值的0.1，AdamW，梯度裁剪1.0。
  相同数据、退化随机流、D5a噪声种子和更新数；逐步输入摘要核验实际配对。
- 沿用现有实例核心与像素语义损失。B粗语义输出插值到GT网格后计算损失。
- 固定第60轮作最终权重，不以训练损失或真实测试图目检选择epoch。

D5a固定权重：`outputs/d5a/candidate/epoch_060.pt`，SHA256
`048954795dc634f82fecf96d5e02bb2f2505afe5eb76298dff5a1b0713b3c80a`。
原S-align/V的来源checkpoint、epoch与摘要随新bundle保存。冻结检查包含LoRA参数以及
encoder、affinity的全部buffers，不能只核对SAM2基础权重。

## 执行与产物

```bash
python tools/run_semantic_d5a.py --config config/train/semantic_d5a.yaml --smoke
python tools/run_semantic_d5a.py --config config/train/semantic_d5a.yaml
```

入口配置`config/train/semantic_d5a.yaml`，单组训练器`train_semantic_d5a.py`。
队列先A后B，任一失败即停止后续。短测每组2次更新，包含保存、严格重载、固定四图
最终推理与对照渲染；正式结束后自动推理复赛100图、校验/打包两组并渲染。
不上传竞赛平台。

短目录：`outputs/semantic_d5a/`，短测为`outputs/semantic_d5a_smoke/`：

- `full/`、`simple/`各有`status.json`、`steps.jsonl`、`epochs.jsonl`、
  `last.pt`和最终`epoch_060.pt`；不累计保存60份完整模型。
- `monitor/epoch_NNN/`固定test_101、test_091、test_130、test_089，保存e0、e1、每5轮和最终
  语义／边界缩略图。过程图不充当验证集或checkpoint选择依据。
- `pair_check.json`确认相同冻结模型、数据和实际训练输入；
  `pipeline_status.json`说明队列所在阶段。
- `full/deployment/`、`simple/deployment/`，最终`submission_full.zip`、`submission_simple.zip`，
  `comparison/`与原baseline、原D5a同坐标／同颜色渲染。

affinity概率图保持不变，但原部署中语义会参与后处理和类别投票，所以完整推理的最终实例
划分仍可能变化。不能把本轮结果等同于此前“固定实例PNG只换类别”的交叉实验。

## 验证与状态

2026-09-25 11:09（Asia/Shanghai）正式队列已启动，先A后B，后台队列PID 11707。
训练环境为服务器`sam2_env`，热数据与产物均在`autodl-tmp`；不持续人工盯训。
`launch.json`保存实际命令、启动时间与四个运行源码文件的SHA256，已核对与本地完全一致。

- CPU相关检查28项通过，包括旧无端点配方的逐值兼容、端点场真实应用、GT同步、
  随机初始化范围、LoRA与buffers冻结检查及严格checkpoint加载。
- GPU短测A/B各2更新，零失败，所有语义模块参数有实际更新；编码器、LoRA、affinity及D5a
  状态不变。两组输入摘要、顺序、恢复种子完全配对；初始affinity概率逐值一致。
- 两组保存后严格重载，语义／affinity logits最大差均为0；同模型训前后affinity输出逐值不变。
- A初始test_069、test_101的最终uint16实例图和类别映射精确复现原D5a部署。
- 短测四图最终推理、uint16检查和对照渲染全部完成，过程缩略图已检查可读。
- A可训练6,302,010参数，B可训练6,038,401参数；含D5a的总量分别84,951,045与84,687,436，
  均低于500M。短测峰值已分配显存约2935／1650MiB，仅作工程资源参考。

正式训练尚未完成，没有本轮官方成绩；上述检查只证明实现与执行链可运行。

同日后续用户回报[既有分类交叉包](SEMANTIC_CROSS_20260925.md)：D5a划分＋原图分类
0.8445／0.8679（85.620），原图划分＋D5a分类0.8449／0.8440（84.445）。前者成为本轮
完成后的额外实测参照。两种固定划分上D5a分类均mIoU微升、面积项下降，支持优化输入与
分类的配合，但尚不证明重新训练必然有效。本次只补分析参照，不改正在运行的训练配方。
