# SSL历史参数补证：Git早期历史与fs归档

核对日期：2026-11-01。目标是当前训练谱系最初的SAM2 Hiera B+ LoRA自监督预训练，
不是后来的Hiera-L、ImageNet归一化或direct-SSL监督实验。本次未训练、未推理、未调用GPU，fs归档只读。

## 1. 对此前结论的修正

此前只读取短`meta.json`与随包源码，没有展开原SSL `best_lora.pth`；据此把LR等一概归为默认值，证据搜索不充分。
本次从Git早期历史和fs归档补到了原checkpoint的优化器状态，并与目标导出权重完成绑定。
**LR、AdamW参数、15000次更新已获实物确认；batch强烈指向2，但原启动命令及实际mask ratio尚未找到。**

| 项目 | 目前结果 | 证据强度/限制 |
|---|---|---|
| LoRA结构 | rank16、alpha32、注入48层、1,019,648参数 | 原meta与checkpoint；注入数量不代表全部实际更新 |
| 训练轮数 | 30；best的epoch字段29 | 原meta＋原best |
| 优化器 | AdamW，两组LR均1e-4、betas(.9,.999)、wd1e-4、eps1e-4 | 原best保存的优化器状态直接确认；LR是保存时值，历史源码无日程 |
| 更新数 | 16个有状态参数的step均15000 | 原best直接确认；无逐步轨迹 |
| batch | 恒定1000源、30轮、固定批量、无额外优化器历史时为2 | 条件性强推断；同期README写8，与更新数冲突 |
| mask ratio | 默认.4，实际命令覆盖值未知 | 历史源码默认；checkpoint未保存CLI参数 |
| 完整SSL训练seed | 未恢复，也未发现历史入口显式固定它 | 不能用config中的数据划分seed42替代；未保存完整RNG状态 |
| 视野/损失/精度 | 1024 letterbox、32–64px随机块遮挡、遮挡区域RGB L1、FP32、clip5 | 原best config的尺度＋历史源码；未恢复执行日志 |
| 完整命令、逐轮耗时、是否续训/中途改参数 | 未恢复 | 需要原运行机器的shell、终端或训练日志 |

这里的“有状态参数”指优化器为曾获得梯度的参数保存了更新计数和历史量。
它能直接证明保存时的优化器设置，不能代替全部训练过程日志。

## 2. checkpoint与目标导出的实物绑定

| 文件 | SHA256 | 大小与核验 |
|---|---|---|
| `outputs/lora_pretrain/lora_state_dict.pth` | `4701e770e658a4f152da695c4b3a3dbb3a0b66152d63c03626399cfea2f4c152` | 4,110,259字节，fs原归档成员 |
| `outputs/lora_pretrain/best_lora.pth` | `650a77ed1ee99c0c36d60072b1b8b6a167d5df9b8a483c42538f644ac866e368` | 7,633,163字节；当前服务器副本与fs manifest同SHA |
| `outputs/lora_pretrain/meta.json` | `fe842d82e591ef705c053cc46e2a5fd94d626a44b9b3e744c58b1face7498bae` | 135字节，best_loss `0.2653634955137968` |

best的96个LoRA张量与4701导出的键集合和每个张量完全相同。
两份LoRA状态的统一张量摘要均为`b24aa7fc7887edbc586a1cac8032d9b67bc6a8621375cd1534511a62fe90403b`。
因此不是拿另一场SSL的优化器参数代替当前谱系。

best中的config是共享基础配置，**不含SSL argparse的完整args**。
其中`data.batch_size=4`、监督训练LR等字段不由SSL对应参数读取，不能用它们覆盖实际优化器或推定SSL batch。

## 3. batch8文档与15000更新的冲突

最早SSL源码提交为`f5ce08d2e797e67e01dbafe6ab9b98cfca458bbc`（8月4日），默认batch2。
其DataLoader `drop_last=True`，每个batch执行一次`optimizer.step()`，无梯度累计、AMP跳步或学习率日程。

8月5日`f80eea6f5d93fb87a80824d62a3b04d112041cd5`的README第106行写：

```bash
python tools/pretrain_lora_ssl.py --config config/default_config.yaml --epochs 30 --batch_size 8
```

8月31日`c54b1be489d7a818f8022a24dd6602123e183097`的`repro/train.py`再次采用batch8；
同提交`repro/README.md`绑定4701目标SHA与30轮。这是文档/恢复配方，不是保存下来的原运行argv。

按1000源、30轮计算：

- batch2：`floor(1000/2) × 30 = 15000`，与原优化器一致。
- batch8：`floor(1000/8) × 30 = 3750`，与原优化器不一致。

10月3日数据归档manifest直接列出1000个唯一无标签源，全部有内容SHA，确认了历史快照中的1000源规模。
但该快照不能证明8月4日run期间从未改变数据或批量；入口也允许加载旧优化器续训。
所以把batch2写作条件性强推断，保留文档冲突，不把它提升为“原CLI已确认”。
旧`repro/train.py`本次不改；后续编排重训时必须显式选择配方并标注恢复依据，不能宣称照旧入口已经忠实复现。

## 4. 额外发现：48层注入不等于48层SSL更新

原优化器的LoRA组注册96个参数张量，只有编号0–7的8个有Adam状态；重建头编号96–103的8个全部有状态。
16份状态的step全部15000；其余88个LoRA参数没有优化器状态。

最早源码用`head(feats[0])`进行重建，只取编码器最浅层输出。
这对应前两个Hiera block的qkv/proj共4层LoRA、8个参数张量，共21,504个标量参数有Adam记录；
重建头为277,379参数。不能把它写成全部1,019,648个LoRA参数均获得SSL更新，
也没有与随机初始化逐张量比较来证明其余88个张量在所有历史环节都绝对未变。
因此恢复时须保留这条历史计算路径，不能因“注入48层”就描述为所有层都经SSL优化，
也不能为了让所有层都有梯度而静默改成多尺度重建。此处没有重放逐步梯度，也没有据此评价SSL对最终官方得分的贡献。

## 5. 本次实际搜索范围

### GitHub对应Git历史

- 当前仓库非浅克隆；检查全部358个可达提交及4个历史SSL脚本版本。
- 只读核对远端公布的5个分支尖端，均已在本地对象中；未发现额外tag，未fetch、切换checkout或修改refs。
- 追查原目标SHA、best_loss、SSL启动参数、README、repro与早期配置，没有找到目标run的原始argv/逐轮日志。
- 最早源码SHA256为`d195efbff4690851cb5a6ad019739255b51ea5d4d0ddf8ed4a87c640403580b1`。

### fs盘与当前服务器

- 只读核对`/root/autodl-fs/private_archive/fast_20261003/`、`fast_20261009/`共15份manifest。
- 在`fast_20261009/legacy_2.tar`定位原best、目标导出和meta；仅读取所需小文件，不整包展开或移动归档。
- 从`fast_20261003/segmentationv2_main.tar.gz`只读README与SSL脚本两个文本成员，逐SHA核对manifest。
  源码SHA256 `20ac3267af59305a4473c60d6ebfe864e812c49dbfcdddb271334b4bf5919cd0`，
  对应Git blob `75740253503d74b6bc341e97f825c336cf9c4c08`，首次见于8月18日`5e201b4b…`；
  相较8月4日`4099298f…`版，仅配置加载和路径解析改变，优化器、训练循环、默认值及特征选择相同。
  这是10月封存源码，不冒充8月实际执行源码；封存README的blob首次见于8月21日，亦不冒充原启动记录。
- 数据manifest SHA `06725c8ed55cbb645ac8f45448cd35aeacfcb0a77dd6e2c10d6b6d439fad71c4`，
  只数条目与校验字段，未读取原图内容。
- 当前服务器同SHA的best通过CPU读取；未实例化模型。当前root的bash/zsh历史文件不存在。
- 这些manifest中没有原`lora_pretrain`命名的启动日志或shell历史；其他变体日志不属于本目标证据。
  **没有逐字搜索fs所有通用日志内容，也没有连接两台尚未开启的历史服务器。**

## 6. 还值得从历史服务器找什么

优先8月4日实际训练这份B+ SSL的原实例，另一台暂不必同时开启。
重点是原用户shell历史、`nohup.out`/训练stdout、终端保存、启动脚本，以及可能尚存的逐轮checkpoint：

1. 原完整命令中的`--batch_size`、`--mask_ratio`、`--resume`；
2. `dataset: ... images, ... steps/epoch, batch=..., lr=...`日志；
3. 前后checkpoint的epoch与step，排除中途换批量/数据或继承额外优化器历史；
4. 当时未提交的源码、环境版本和起止时间。

若两台都不是8月4日原训练机，优先检查哪台保留原home/系统盘；仅有已同步到fs的输出副本，新增信息可能有限。
即便找不到固定seed记录，也不能声称原运行设过完整seed；未来重训应明确新设并保存RNG状态，与历史恢复分开说明。

## 7. 私有证据索引与验证边界

以下均位于被忽略的`output/ssl_parameter_search_20261101/`，不进入Git：

| 回执 | 内容 |
|---|---|
| `git_history_receipt.json`、`exports_manifest.json`、`script_blob_variants.json` | Git只读范围、原文、历史版本；单线初查里的“LR未知”已由本次checkpoint补证取代 |
| `fs_checkpoint_receipt_v2.json` | 原best/导出/meta的SHA、逐张量一致性、优化器状态 |
| `optimizer_step_interpretation.json` | 独立复核batch算术、前4层LoRA的状态映射与源码版本差异 |
| `fs_source_v2.json`、`fs_source_ascii_receipt.json` | fs源码与Git blob绑定、1000源计数、原优化器分组、阴性搜索边界 |
| `fs_findings.json`、`fs_inventory.json`、`fs_text_candidates.json` | manifest定位、定向文本候选和阶段性汇总 |
| `live_result.json` | 当前实例历史文件存在性与LoRA目录小范围只读检查 |

本次只更正[当前训练记录](TRAINING_RECIPE_20261101.md)及新增本报告，未修改训练代码、现有权重、GT或提交包。
参数补证缩小了历史缺口，仍不能替代当前完整训练链的从零实跑验证。
