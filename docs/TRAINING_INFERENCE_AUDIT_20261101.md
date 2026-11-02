# 最终训练与推理逻辑审计

2026-11-01。用户要求审计实际训练、推理可用性，使用`dualog-audit`安排Claude Code独立审查，Codex负责复核、修复和测试。本报告不将CPU模拟、历史GPU证据和本轮真实执行混为一类。

## 审计状态与范围

Claude会话：`dialog-1793518256291-22e2ddb2`，`partner_agent=claude`、服务端默认模型、回显`effective_reasoning_effort=high`。未显式指定模型、轮数或effort。**Claude独立审计未完成：完成0轮，未收到任何审计发现或通过结论。**

15:46:25（北京时间）运行器确认失败并退出：`Claude headless turn exceeded 900000ms with no completion`。
末尾输出同时为`[claude-code:unrecognized_model] {"model":"deepseek-flash[1m]","query_source":"sdk"}`。
这说明调用超时并伴随SDK模型识别提示，尚不足以单凭该行确定配置或服务端根因。按skill的失败处理要求停止等待并关闭会话，不自行切换用户模型或重启。以下是Codex与本地子代理完成并验证的修复，不冒称Claude审核通过。

覆盖当前实际调用链：

- `train_pipeline.py`、`repro/pipeline.py`、`current_early.py`、`current_backend.py`、`current_naf.py`及默认/教师配置；追踪SSL、Stage1、joint、S-align、V、D5a、R1、缓存、NAF和导出。
- `infer.py`、`infer_compact.py`、`infer_zero.py`、`infer_flat_semantic.py`、`infer_boundary_rgb.py`、`infer_semifinal.py`及实例读出、投票、面积整理。
- `infer_submission.py`、`build_semifinal_bundle.py`及新训练产物桥`infer_trained.py`。
- `infer_accelerated.py`、`tensorrt_runtime.py`、`build_inference_engines.py`及相关配置/依赖/测试。

审计不读取测试GT、不修改监督数据、不启动训练、不轮询后台150任务；原冻结任务快照保持独立。跳过无关历史实验、二进制和压缩包，不将赛题图像或派生数组纳入公开报告。

## 已确认并修复

| 问题与触发 | 实际影响 | 修复与验证 |
|---|---|---|
| 公共训练链导出三权重后，默认推理仍锁原C；公开旧打包器不提供当前三模型桥 | 新训练结果需额外私有脚本才能按现有C算法推理 | 新增`infer.py --training-manifest`；直接核本轮导出清单、三文件SHA、状态数量、格式、NAF合同及500M上界，记录独立模型身份，作用域退出恢复旧C保护 |
| 便携入口在内层已成功后，再因源码或模型收尾校验失败 | 命令报错但既有报告可能仍`complete=true` | 将执行与全部收尾纳入失败处理，撤销成功状态并写`submission_failed.json`；覆盖四种收尾/执行失败 |
| 正式训练环境没有CUDA，部分旧训练器会自动退到CPU | 可先慢跑多个阶段，后续强制CUDA时才失败 | 在真实GPU阶段启动前及内部dispatch检查CUDA，明确提示`sam2_env`；CPU导出、显式独立导入、joint接续、只读命令及模拟executor不受限制 |
| 默认原GT Stage1未配置过程监控，教师配置单独配置掩盖了遗漏 | 新fresh训练不会按约定保存固定训练缩略图 | 默认配置补e0/e1、每10轮、最后一轮的384缩略图；通过真实monitor CPU e0输出核验，不改变训练采样与预算 |

新权重推理命令：

```bash
python infer.py --backend torch --training-manifest outputs/train_run_01/verify_export/manifest.json --input-dir data/test/semifinal --output-dir outputs/infer_trained_01 --dataset semifinal_150
```

三权重由清单提供，不另传`--model-dir`或`--naf-checkpoint`。目前仅Torch；与TensorRT组合会明确拒绝，不能把新权重配上旧C引擎。它是本地训练产物的推理桥，不自动晋级主线，也不是完整提交包。

## 本轮验证证据

1. 修改前扩大范围的18个相关测试文件共296项通过，作为基线；修复后加入新入口、CUDA检查、失败收尾及过程图测试，20个相关文件联合**345项通过，51.02秒**。测试覆盖实际小CPU状态导出、真实配置校验和monitor，以及模拟GPU/引擎调用；不是全项目所有测试或实际GPU执行。
2. 默认13阶段代码预检通过，未加载模型、启动训练或创建run目录。根入口新权重路由及旧Torch/TensorRT帮助命令通过。
3. 在服务器使用已完成教师演习的**真实三份训练导出**运行新`validate_manifest`及有效配置校验：通过，状态元素参数上界102,583,335，协议`explicit_zero_v1`，CUDA未初始化、没有模型前向。私有回执为`output/logic_audit_20261101/real_exports_cpu.json`；这证明真实导出结构可被接受，不证明新入口已完成GPU推理。
4. 独立源代码复核确认：分散权重统一以训练run作相对路径根，符合原报表`relative_to(model_dir)`要求；NAF自己的状态摘要与通用tensor摘要分开；成功和失败退出均恢复临时方法与原C身份锁。

完整操作见[使用指南](TRAINING_INFERENCE_GUIDE.md)。

## 仍然独立的验证边界

- 本轮未从原GT执行完整fresh GPU训练，不能据此承诺重现原C权重或官方成绩。SSL历史参数恢复缺口仍按[原记录](SSL_PARAMETER_RECOVERY_20261101.md)保留。
- 新整理的公共引擎构建工具及新训练推理桥尚未实际GPU实跑。先前私有构建/整图GPU结果见[全模块记录](FULL_ACCELERATION_20261101.md)，不能转记为这些新入口的GPU证据。
- 既有全五引擎150图任务仍由用户通知完成后取包、独立回分；本次代码修复不覆盖该封存版本。
