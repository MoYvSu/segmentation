# 当前三模型训练链统一入口

状态更新：2026-11-01。顶层训练入口为`python train_pipeline.py`，直接转发既有`repro/pipeline.py`；
默认配置`repro/current_pipeline.yaml`使用原人工标注/purified GT及后期r8，默认13阶段。
**这份原GT配方尚无从头完整GPU实跑记录，也未证明可重现已评分原C的权重或得分。**

[教师GT监督链](TRAINING_RESULT_TEACHER32_20261101.md)已经跨两段run执行完毕，最终exit0，
端到端5小时46分23秒；其配置为`repro/teacher_pipeline.yaml`，SSL/D5a显式导入而未重训。
随后[完整150图推理及用户回分](TEACHER32_C150_INFERENCE_20261101.md)完成：mIoU .7106、面积 .8975，
等权参考80.405，低于原全零C的82.660，候选淘汰。工程执行成功不等于效果通过，也不等于原GT冷启动复现成功。

默认链已[删除V6与未使用的语义边界缓存](V6_ELISION_20261101.md)，S-align直接从joint构建固定语义教师；
旧15阶段仅在`semantic_teacher_source: legacy_v6_e9`时启用。旧`repro/train.py`仍是E10a/G4b历史链。
当前推理另由顶层`infer.py`选择`--backend torch|tensorrt`，默认`torch`；训练入口不自动启动推理。
训练结束可显式传`infer.py --training-manifest <run>/verify_export/manifest.json`接入本轮三权重，
保持全零C算法并另记模型身份；这一路目前仅支持Torch，旧C默认及旧引擎来源保护保留。
GPU计算阶段已加CUDA启动检查，CPU导出/显式导入和只读检查不受限；默认Stage1固定训练缩略图已接通。

历史实施经过保留如下，均不表示任务目前仍在训练：

- 初版只实现入口、CPU和资源校验，当时未启动训练或生成GT。
- [教师32源演习](TEACHER_C32_REHEARSAL_20261101.md)随后启动；删除V6时保留正在执行的joint，完成后通过`completed_joint_continuation`及三个源SHA在新run接续，没有重复Stage1或伪造旧身份。
- 04:29的[夜间检查](TRAINING_NIGHT_AUDIT_20261101.md)修复教师NAF第0步过程图并部署接续v3；取消无关的全局GPU空闲、旧队列退出码和12小时截止要求，来源与资产检查保留。

## 1. 覆盖的实际训练链

```text
现有purified GT + 原32 LabelMe + 1000训练图 + 官方SAM2
  SSL -> Stage1 -> Stage1边界缓存 -> joint-v3
    joint -> 固定语义教师 -> S-align（学生语义完整随机重置）
    joint -> V affinity（不重跑G0/G1/G4b）
  1000训练图 -> 独立D5a e60
  joint + S-align + V + D5a -> fresh R1 mixed balance -> backend/restorer导出
  本轮backend/restorer -> 训练32源局部零噪N1缓存
  只读GT32封装 + 新N1缓存 -> NAF width1000 -> NAF G导出 -> 三模型身份/参数预算校验
```

默认共13个阶段，V与S-align是独立训练分支。旧语义缓存/V6仅供显式legacy兼容。既有SAM64沿原清单与授权复用；不重新生成几何候选或捏造旧approval。
所有已有GT作为显式输入，本入口不运行LabelMe净化、r8补缝或teacher-GT修订。
教师演习使用单独的teacher格式及监督域，不能与原r8混写；未来GT版本需另行固定来源与有效域。

SSL仍保留原浅层重建路径，未顺带实施深层SSL候选。
batch2是15000步的条件性强推断，mask .4是恢复默认值；新固定随机流不冒充历史原seed。
原data split seed42保留，V的初始化/划分/采样seed也整体保留42。
详细预算、冻结及来源见[训练参数总表](TRAINING_RECIPE_20261101.md)和[SSL补证](SSL_PARAMETER_RECOVERY_20261101.md)。

## 2. 只读检查与演练命令

在项目根目录、`sam2_env`训练环境中使用。下面三条命令都不启动训练：

```powershell
python train_pipeline.py list
python train_pipeline.py check --code-only
python train_pipeline.py run --dry-run --config <私有资产映射.yaml> --run-dir outputs/repro_current/rehearsal_v1
```

`plan`还会列出每阶段前置关系与新输出路径。`check`核验现有输入，不要求尚未训练的中间权重存在。
`--code-only`只查入口/配置，不能当作资源校验通过；`run --dry-run`默认同时做资源预检。
资产报告可用`--report output/<独立目录>/check.json`保存，逐源清单仅写私有目录。

公开配置使用项目相对路径。原图、GT或SAM64存放位置不同时，另写被忽略的资产映射，例如：

```yaml
_base: ../../repro/current_pipeline.yaml
assets:
  sam2_geometry_dir: output/delivery_training/resources/data/sam2_geometry_g2
```

该示例假定映射文件位于`output/<目录>/`；`_base`相对于映射文件解析，资产路径相对于项目根目录解析。
本次本机资源映射为`output/repro_pipeline_validation_20261101/local_assets.yaml`。
不把本机/服务器绝对路径写进公开配置，不复制或移动原数据。

## 3. 正式执行与产物复用

需要新训练时，使用同一入口去掉`--dry-run`，指定全新的`--run-dir`。本文整理不启动新训练；既有教师演习见首节记录。
`--from-stage`/`--to-stage`支持分段执行；只执行后段时，前置阶段必须已有本轮真实完成回执。
R1直接训练已采用配方，不要求先训练历史control/native/mixed旁支。

每个阶段都有“完成回执”：记录实际输入、源码/配置/环境身份、输出SHA及耗时。
`--reuse-completed`只有在下列条件全部相同时才复用；仅发现一个checkpoint文件不会跳过训练：

- 同一run的配置、资产内容、源码、Python及关键依赖版本；
- 实际上游完成记录与输出内容；缓存同时包含概率数组、names/exclude等索引；
- 本阶段完整输出与adapter的完成结果。

每阶段执行前后再次核对代码和训练输入，变化时保留已有文件并拒绝认证。
中断或失败产物不会自动覆盖，也不会被误认为完成。**统一入口尚未实现单阶段中断后的自动续训**；
R1的`last_head.pt`仅是保存下来的训练状态，不承诺可由本入口直接恢复。
NAF底层支持严格同身份断点恢复，但不把它扩写成整条链的通用恢复能力。

阶段输出均在本轮私有run下；输入GT只读，训练配置写入`configs/`，日志写入`logs/`，完成记录写入`receipts/`。
固定小训练样本缩略图继续保存，不需要代理盯训练，也不会自动运行150张测试推理。

## 4. 新上游的正确衔接

- S-align默认从本轮joint捕获固定教师，再随机重置完整语义头；V从本轮joint边界FPN开始。
  CPU核验joint/S-align共享LoRA一致，组装时继续严格核语义/几何合同；显式legacy才额外核V6 e9。
- fresh R1保留20轮、每轮64更新、32 whole/32 native1024、negative_weight=1.0；只更新affinity。
  保存本轮真实抽样计划、源文件身份、冻结核验及严格重载结果，不拿旧消融逐步日志冒充证据。
- R1训练内D5a恢复保留历史随机工况；当前部署N1缓存使用显式zero。没有把推理全零追溯写成历史训练全零。
- NAF缓存绑定本轮backend/restorer文件SHA、32源GT内容与零噪/融合合同；新GT或新上游不能复用旧13b3/9315缓存。
- NAF使用原width配方：1000更新、batch16、G日程分母53000、kernel8、FP32、GAN .01、前64步无对抗项。
  GT封装只引用现有原数组；未知域继续ignore，预测边界只作输入。
- 导出只保留推理所需状态；D5a保持标准`model`字段，backend/NAF保持各自标准格式。
  最后核三模型文件及状态元素的保守参数上界<500M。

`train_stage2.py`仅增加可选`semi_supervised.checkpoint_epochs`，默认空列表不改变旧行为；
仅显式legacy链指定[9]以额外留下V6 e9，原周期及monitor频率保持；新默认链不运行V6。

## 5. 验证记录与明确边界

首次入口验证完成原15阶段顺序与衔接、现有资源及CPU模拟检查；随后13阶段删减新增76项联测和真实checkpoint CUDA桥接，详见首节链接。以下保留首次实现记录。
实际32源GT通过旧NAF目标读取函数的尺寸/有效域检查，并生成新的只读路径封装；原GT数组未修改。
测试覆盖错误输入、缓存索引篡改、跨配置复用、半途失败保留、R1抽样/冻结、严格FP32导出和NAF缓存/训练合同。

本机资源预检需要上述私有映射；公开默认的`data/sam2_geometry_g2`在本机不存在，不能声称默认路径无需映射即可通过。
校验回执保存在`output/repro_pipeline_validation_20261101/`，不进入Git。

上述首次CPU/资源检查不是完整GPU冷启动证据。后续教师链实跑已完成，但导入SSL/D5a且回分较低；
**原GT13阶段从头完整GPU训练、原C权重及得分重现仍未验证。**
当前已评分compact/zero推理入口仍有旧NAF身份约束；新训练导出通过标准loader的身份校验，
并不表示可以直接替换已评分部署。教师模型已通过独立入口完成推理和评分，随后淘汰，旧三权重保留。
本文整理不启动GT优化或新训练，不改部署默认，不提交/推送私有产物。

## 6. 与推理入口的关系

`python infer.py --backend torch`默认使用原C的`tools/infer_compact.py`；
`python infer.py --backend tensorrt --engine-manifest <清单>`显式选择全五引擎＋CPU整数读出候选。
后者要求每个engine的文件SHA及来源模块state SHA一致，FP32且不启用FP16/INT8，不接受缺角色回退。
编码器/语义/affinity/D5a引擎是静态B1×1024，仅NAF窗口支持batch1–16；150图任务不等于网络batch150。
整合候选的三源短测与新进程预检已通过，150图任务已启动，尚未在此记录完成或回分。
完整参数和命令见[推理接口](../repro/INFERENCE_INTERFACE.md)及[加速报告](FULL_ACCELERATION_20261101.md)。
