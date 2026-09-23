# D4：在线平滑空间模糊与同预算对照

2026-09-23。用户授权先加入空间模糊并开始训练；短链监督暂不实施。
承接[逐步收益诊断与方案](RGB_DIFFUSION_D3_TRAJECTORY_20260923.md)。

已于北京时间19:49:58启动顺序任务，队列PID30238，代码版本`6d6be7e`。
先执行D4，再执行原D3配方续训对照。两组均从D3第20轮／5000更新状态继续到
累计第60轮／15000更新，各新增40轮／10000更新。使用RTX4090、`sam2_env`和autodl-tmp快盘。
启动后只核验任务正常运行，不持续监控训练；程序自动保存过程图。

启动核验完成：D4训练PID30239，累计5812更新（新增812次、第24轮中），失败更新0。
2597个模糊样本中1282个启用空间增强，实际比例49.36%；起点抽样比例50.09%。
初始17张PNG已生成，模型／优化器／scheduler／scaler／随机状态与父e20逐项一致，
原固定配对逐值相同。原D3对照仍在队列中，尚未开始；核验后不继续轮询。

## 唯一训练配方变化

配置：`config/train/rgb_restoration_diffusion_d4_spatial_all60_monitored.yaml`，继承D3。

| 项目 | 原D3对照 | D4 |
| --- | --- | --- |
| 原始图像 | 全1000张、无留出 | 相同 |
| 清晰／模糊概率 | 20%／80% | 相同 |
| 模糊样本中的空间模糊概率 | 0 | 50%，约占全部样本40% |
| 高斯模糊参数范围 | 0.4–5.2 | 相同 |
| 噪声kappa／起点训练比例 | 0.03／50% | 相同 |
| 模型／损失／原局部任务 | D3 | 相同 |
| 起点／优化器／学习率日程 | D3 e20完整状态、原60轮日程 | 相同 |
| 新增更新数 | 10000 | 10000 |

在等比缩放后的有效整幅内容上、裁块之前生成3～5格的随机网格，平滑放大成空间变化场。
在方差sigma平方上去均值并按上下界余量缩放，使有效整幅内容的平均sigma平方仍等于
原抽取sigma平方。然后在线计算最多8档高斯模糊结果，按局部方差插值，流式累加以控制内存。
最后沿用原来的重采样、反射填充、裁块和翻转，以及裁块后的原局部损伤任务。

插值是空间变化模糊的近似，不声称精确模拟光学焦面。基础sigma接近上下界时，
可用空间变化幅度相应减小；不另外提高模糊上限。
新增随机流使用`seed/epoch/index/6`，不移动原强度、重采样、裁块或翻转随机流。
目标清晰图、有效区域和原mask几何保持一致。缺省关闭、清晰样本及旧holdout路径保持兼容。

## 明确分叉，保留原状态

新入口`--fork-spatial-from`仅允许在新空目录中新增`degradation.spatial_blur`及其独立monitor；
其他配置变化拒绝，普通`--resume`仍不允许修改训练配方。
父权重、优化器、scheduler、scaler、随机状态、累计轮次／更新数和原固定配对全部保留。
新目录先保存带来源记录的第20轮状态，新增训练日志从第21轮开始，不复制父日志伪装新更新。

父checkpoint：`outputs/rgb_restoration_diffusion_d3_terminal50_all60_monitored/epoch_020.pt`。
SHA256：`152411aa656d6e042ca4aa73176a1c5f3f601e50cc3ef90ebd45500c2e23c722`。
原D3的`last.pt`已核对完整载荷与其一致，其文件SHA256为
`fb516f41c2c528cc480e2f489366431a7593a67b1373e2939427cc10eb93efdb`。
对照启动前再次检查其未被其他运行改写；原`epoch_020.pt`始终保留。

```bash
python train_rgb_diffusion_restoration.py \
  --config config/train/rgb_restoration_diffusion_d4_spatial_all60_monitored.yaml \
  --device cuda \
  --fork-spatial-from outputs/rgb_restoration_diffusion_d3_terminal50_all60_monitored/epoch_020.pt

# D4正常完成后，队列依次执行同预算原配方对照。
python train_rgb_diffusion_restoration.py \
  --config config/train/rgb_restoration_diffusion_d3_terminal50_all60_monitored.yaml \
  --device cuda \
  --resume outputs/rgb_restoration_diffusion_d3_terminal50_all60_monitored/last.pt
```

两组均不传`--stop-after-epoch`，按原60轮完整日程结束，不重置为额外60轮。
队列任一训练失败则停止后续任务，状态与退出原因保留在控制目录。

## 过程图与后续比较

原4组训练配对从父运行直接继承，继续记录首步和16步。
另从相同4个训练源图生成固定空间模糊配对：整幅有效内容上的基础sigma取3.0，
平滑变化后中心裁块，不叠加重采样或局部损伤。只作训练过程诊断，不用于选权重。
按各裁块基础sigma下／上三分位划分相对弱／强区域，不能将其当作实际工况的绝对模糊等级。

自动保存内容：

- 清晰目标／空间模糊输入／固定色标sigma图／首步／16步，上排裁块、下排相同中心放大区域。
- 全部16次清晰图预测的RGB、梯度和MSE误差，区分整图及相对强弱区域，并保存曲线。
- 清晰输入经过模型后的首步／16步对比及绝对漂移误差。

D4节点为起点e20及e25、30、35、40、45、50、55、60，每节点原monitor 8张，
空间monitor 8张配对图加1张曲线。首个节点应有17张PNG。
原D3对照继续保存其原配对过程图，训练后再使用D4同一组空间配对作固定输入比较。
不能将D4 e60仅与D3 e20比较后，把全部改善归于空间增强。

## 验证与私有入口

CPU测试共89项通过：数据新增15项与既有28项；训练既有24项、分叉21项、空间monitor 1项。
覆盖关闭逐值兼容、独立随机流、目标几何一致、常量保持、sigma范围和平均方差、
checkpoint完整继承、严格变更边界、连续／分段续训逐值一致，以及monitor不改变训练随机流。

GPU真实数据单步前向／反向通过：1000训练源图、无留出，loss0.023254、梯度范数0.094903，
继承学习率0.0001505，峰值显存分配7743MiB。这只说明增强与训练通路可运行，不代表效果改善。
32个样本中27个模糊样本、13个启用空间增强；正式运行记录每批与每轮实际数量。
样例目检中空间变化连续，没有明显拼接接缝。

服务器工程：`/root/autodl-tmp/segmentationv2_semifinal_20260921`。
新增训练产物：`outputs/rgb_restoration_diffusion_d4_spatial_all60_monitored/`。
队列、日志、启动检查：`outputs/20260923_diffusion_d4_spatial_control/`。
本地对应私有入口：`output/20260923_diffusion_d4_spatial_control/`。
图像、逐图记录和权重均位于忽略目录，不进入Git；未生成提交包或取得新黑盒结果。

命名更新：当前D4可直接通过短入口`outputs/rgb_restoration_diffusion_d4/`访问，
原D3对照入口为`outputs/rgb_restoration_diffusion_d3/`。两者指向原目录，未移动运行中的文件。
D4配置的默认输出目录已改为短名；本次进程仍使用启动时加载的原配置，历史记录不回写。
后续新运行目录最多四段，具体增强／训练预算写入配置与记录。

- [空间模糊样例](../output/20260923_diffusion_d4_spatial_control/augmentation_00.png)
- [GPU预检查](../output/20260923_diffusion_d4_spatial_control/preflight.json)
- [启动记录](../output/20260923_diffusion_d4_spatial_control/launch.json)
- [启动后核验](../output/20260923_diffusion_d4_spatial_control/startup_check.json)
