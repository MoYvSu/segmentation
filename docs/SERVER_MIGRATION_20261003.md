# 2026-10-03：复赛实验资源迁移

用户提供今天的有卡服务器`connect.bjb2.seetacloud.com:15930`，旧服务器`connect.bjb1.seetacloud.com:23411`以无卡模式保留。
必要资源已经迁至新服务器，独立项目目录为`/root/autodl-tmp/segmentationv2_semifinal_20260921`。
本轮没有修改旧资源、覆盖新机其他项目、创建Git仓库、提交或推送代码，也没有启动新训练。

| 分析／迁移方法 | 实际结果 | 结论 |
| --- | --- | --- |
| 枚举当前R1、D5a、coverage和下一次同源训练的必要依赖 | 原5904文件共5,759,729,587字节；追加父缓存清单所需2文件156,358字节 | 当前必要资源范围5906文件，约5.76GB；不搬无关旧实验和中间权重 |
| 只读复用新机同内容资源，其他文件经共享FS中转 | 1711文件从新机旧副本复制；其余通过临时载体迁移，逐文件SHA验证 | 最终数据、权重及源码实体在新机临时盘，推理／训练不持续读取FS |
| 检查实际数据、规范与缓存闭合 | 1000训练图、100测试图、32人工原图和完整processed新GT、64允许伪标注源、32原R1图缓存 | filled参与，GT0和padding忽略；父capture清单99成员全部精确一致 |
| 核对环境与源码 | Python3.12.3、torch2.8.0+cu128、CUDA12.8、numpy2.3.2、cv2 5.0.0；SAM2源码／配置／HEAD和实际解析路径一致 | 无须安装或升级环境；原49、148、157份封存源码与当前157依赖通过SHA检查 |
| 独立重读正式训练状态 | 原`verify_run`确认e20／1280、39个优化器状态、学习率／随机状态、最终张量及严格重载身份 | 完整训练已经结束，迁移不等于重新训练或性能增益 |
| 临时载体清理与可用资源 | 本轮四个FS迁移／分析中转目录已清理，两个分析导出释放115,942,034字节；新tmp约13GiB可用，RTX4090可用 | 保留原服务器、原成绩包、完整本地分析和回退权重 |

## 当前入口与权重

训练／推理环境使用`/root/miniconda3/envs/sam2_env/bin/python`。
分析入口为`tools/analyze_affinity_coverage.py`；新分析脚本独立于原157份训练快照，不改写训练／后处理。

| 资源 | 新项目内路径 | SHA256 |
| --- | --- | --- |
| 原R1 mixed balance e20 | `outputs/affinity_balance/mixed/final.pt` | `43da103c95c7f85149bd0533fd116d26f0a81c91d793db7e8938b7d4ae80e434` |
| 固定D5a e60首步 | `outputs/d5a/candidate/epoch_060.pt` | `048954795dc634f82fecf96d5e02bb2f2505afe5eb76298dff5a1b0713b3c80a` |
| coverage e20 | `outputs/affinity_coverage/candidate/final.pt` | `0f828a71cc8550ef3b00958f88bc3b24e2b9db70d8f79861ec8b949500ad9af0` |

必要初始化链与原whole、mixed、p1024及技术短测参考均按原相对路径可读；coverage保留final、last_head、head20、全部过程图、100图输出及来源回执。
SAM2局部路径链接新机已有临时盘仓库，574工作树文件全部一致；兼容历史`segmentationv2_work`路径的链接仅指向新独立项目，未链接到FS。

## 回执与边界

本地私有回执在`output/server_migration/`。原5904文件清单SHA为
`8e5507cefd847ec33fdfb8099f524d36b55433600f634b9649b695f5656241d3`。
初次GPU完整性检查发现精简选择遗漏父清单中的进度回执与源码副本；补齐原字节后重新核验全部99成员，没有关闭或放宽检查。
原清单和原回执保留，补充验收`dependency_addendum.json` SHA为
`3a7c6464edecd9384f0b9f106a2adfe538a33b88c1226b92090b88eea84f6277`。
独立续训状态验收为`independent_resume_audit.json`；GPU跨机数值复现与候选效果见[coverage分析](AFFINITY_COVERAGE_20261002.md)。
临时目录删除前核对解析后的绝对路径与`/autodl-fs/data`及四个确切目录名；`staging_cleanup.json`／`export_cleanup.json`记录完成，不清理其它FS内容。

本页只记录必要迁移与复现条件，不是技术报告或正式提交材料。全部数据、掩码、对比图、逐图JSON、模型与提交包只在被忽略目录中保留。
