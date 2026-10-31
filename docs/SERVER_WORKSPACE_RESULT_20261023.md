# 服务器双路线工作区整理结果

2026-10-23。按用户授权，先保存权重和未提交源码，再建立有监督、严格无监督两个工作入口。
本轮没有训练、修改模型行为、替换已评分部署，也没有提交或推送Git。

## 当前入口

| 用途 | 服务器路径 | 说明 |
| --- | --- | --- |
| 总入口 | `/root/autodl-tmp/seg/README.md` | 另有`COMPATIBILITY.md`和私有`assets.json` |
| 有监督开发 | `/root/autodl-tmp/seg/supervised/code` | `main`，导入本地实际工作区，含未提交实现 |
| 无监督开发 | `/root/autodl-tmp/seg/unsupervised/code` | `codex/unsup-u0`，严格来源限制不变 |
| 历史实验与控制 | 两条路线各自的`runs/history/` | 24个有监督、43个无监督目录的快捷入口 |
| 权重 | 两条路线各自的`models/` | 标明原训练权重与导出权重；不混为可续训状态 |
| 无监督缓存 | `unsupervised/cache/` | 固定CV、phase、纹理特征、近邻和22视图缓存 |
| 公共资源 | `shared/train/`、`shared/foundation/`、`shared/sam2` | 仅官方训练原图、基础权重和SAM2代码依赖 |
| 已评分结果 | `supervised/release/scored/reflect1024.zip` | 用户报告的最高分0.8645／0.9006，等权88.255 |
| 可移植推理工程 | `supervised/release/portable/code_model/` | 后期噪声复现修改的工程，与历史评分结果分别保存 |

共建立106个链接。所有新链接指向旧实体；旧路径没有改名、搬移或改成反向链接。
保留这一方向，是因为历史预检会解析真实路径并核对源码、缓存及回执身份。

两份代码是**同一个Git仓库的两个worktree**：共用提交历史，分别工作在不同分支。
远程仍配置为原GitHub仓库，本轮从经审查的本地Git bundle导入，没有执行GitHub同步或推送。
Git公共目录位于`supervised/code/.git`，不得把该目录当作独立可删除副本。

有监督基准commit为`7e5842e208fe8a547f7dc30c48c269405adb2ac0`；
无监督基准commit为`56b851ec0b52bcd8cb2bcddb572fb77b2407a178`。
两者均在checkout后覆盖**已核验的实际源码快照**，因此不能只用commit代表当前工作代码。
导入的869和1025个文件分别与各自快照逐文件一致；真实未提交改动保留。
数据、权重、输出入口均受本地ignore和独立保密Git hook保护，暂存区为空。

## 保存内容

服务器归档根：`/root/autodl-fs/seg_archive/20261023/`。

| 保存方案 | 核验结果 | 结论 |
| --- | --- | --- |
| 5份原资产归档完整重读 | 9,456,998,400字节，37,609普通文件及179链接；内容SHA与属性全部一致 | 权重、实际源码、配置、回执和monitor已另存 |
| 全部现存checkpoint归档 | 237个文件、6,576,743,870字节；7个重点权重对齐历史SHA | 不仅保留终态；文件数不代表独立模型数 |
| 标注补档 | raw32、现存purified_gt、SAM64共323文件 | 现存purified版本不自动等同最高分所用新GT |
| 本地3个工作树快照及Git bundle | 主树、无监督、旧扩散的实际源码与差异保存；bundle两分支全部祖先通过保密检查 | 未提交代码不依赖GitHub是否已同步 |
| 原实验整理前后比较 | 3,402路径、2,393个唯一文件SHA：0差异、0错误；整理前4,870项历史期望SHA均一致 | 控制、固定缓存与历史引用未被改写 |
| 新入口CPU验收 | 6配置、4 CLI帮助入口、9 checkpoint安全读取通过；2,105份源码/工程文件前后SHA一致 | 配置解析、CPU读取和入口有效，不代表GPU推理或续训通过 |

原资产分为`shared/assets.tar`、`supervised/assets.tar`、`unsupervised/assets.tar`、
`supervised/annotations.tar`和`metadata/source_additions.tar`。
本地代码另存`local_code/`，评分结果和可移植工程输入另存`release_inputs/`。
新入口清单及验收记录另存`workspace_layout/`。

服务器13个历史Git根的实际状态和源码已保存：8个HEAD正常、4个尚无提交，1个旧worktree元数据指针失效。
没有为整理而修复或伪造这些历史Git状态；实际文件归档不冒充完整Git历史备份。

1000训练原图和部分大型NPY/NPZ缓存仍是外部依赖，详见归档中的`external_dependencies.json`。
这是一份精选资产备份，不是丢失全部原数据后仍可独立恢复所有实验的镜像。
快盘和慢盘也不宣称是异地灾备。

## 后续使用与兼容边界

- 新实验使用各路线`runs/<短实验名>/`；配方、精确源码清单、日志、monitor和权重一起保存。
- 历史控制仍由`runs/history/`进入原封存源码，按原命令运行；不把新开发代码覆盖到旧实验。
- 无监督完整Affinity所需核心源码与历史协议一致，输入依赖仍通过显式CLI传入；不绕过原预检。
- 语义蒸馏须显式指定原教师/控制与新输出。仅补无监督主线索引的语义权重叶子入口，没有把整棵旧output联进新输出。
- 有监督主代码中的`submission_reflect.yaml`仍是导出模板。实际可移植推理入口为`release/portable/code_model/infer.py`及包内README。
- 无监督开发目录未连接人工GT、人工筛选SAM64或有监督训练权重；共用架构实现不表示共用监督来源。
- R2与完整Affinity的权重明确记录`resume_implemented:false`；保存optimizer/RNG并不使续训入口自动成立。

当前服务器无卡，本轮没有模型forward、GPU推理、训练或resume验收。
旧大缓存仍被活跃协议引用，本轮不清退旧实验目录；仅清理本次已经校验冷备的冗余上传暂存。

本地私有回执统一在`output/workspace_setup/`，不得提交Git：
`remote_backup/receipts/backup_receipt.json`、`install_receipt.json`、
`reference_audit/after.json`和`cpu/receipt.json`分别记录备份、安装、旧引用复核和CPU检查。
公开文档只记组织方式和汇总结论，不包含赛题图样、权重、预测或提交产物。
