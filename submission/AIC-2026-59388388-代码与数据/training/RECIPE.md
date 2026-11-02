# 默认训练配方

种子为20261101，D5a为42，R1为20260928，R1 mixed view为20260930。输入为32人工源与r8目标、64份已授权无类别SAM2几何和1000无标签训练源。每阶段实际展开配置保存在新run的configs/。

官方SAM2预训练权重仅严格加载`image_encoder.trunk`；FPN与任务decoder随机初始化。

| 顺序 | 阶段 | 初始化/更新范围 | 默认预算与关键参数 | 产物 |
|---|---|---|---|---|
| 1 | SSL | 官方SAM2底座；LoRA与重建头 | 30ep，B2，AdamW lr1e-4，mask .4，rank16/alpha32 | lora_ssl/lora_state_dict.pth |
| 2 | Stage1 | SSL LoRA + 随机任务decoder | 100ep，B8，512 crop，lr1.5e-4，warmup25后cosine，wd1e-4，FP32 | stage1/best_model.pth |
| 3 | Stage1边界缓存 | 本轮Stage1 | 1000源，1024等比letterbox | stage1_boundary/boundary_probs.npy |
| 4 | joint | 本轮Stage1；语义/边界头 | 100ep，人工B4+无标签B4；250无标签样本/轮；lr5e-5，warmup5/flat50/decay至.4 | joint/best_model_stage2.pth |
| 5 | S-align | 本轮joint为固定teacher；完整语义冷重置，LoRA/边界冻结 | 20ep，每轮62更新，人工B4；lr1e-4，warmup2/flat12/decay至.25，监督+语义蒸馏 | semantic/best_model_stage2.pth |
| 6 | V affinity | joint边界FPN初始化；冻结encoder/LoRA/语义 | 120ep，52样本/轮，B2=26更新/轮；head lr1e-4/FPN5e-5，最低5e-6，AMP；negative1.5/hardnegative1 | affinity/best_affinity.pth |
| 7 | D5a | 随机pixel diffusion NAFNet | 60ep，1000源，B4，512crop；lr2e-4至2e-6，wd1e-4，AMP；16步/kappa.03；RGB1+gradient.2 | d5a/epoch_060.pt |
| 8 | fresh R1 | 本轮joint LoRA + S-align语义 + V affinity；D5a冻结，只更新affinity | 20ep，B1两流各64draw/轮，共1280更新；32whole/32native1024；affinity lr2e-5降至.1倍，wd1e-4，FP32；人工loss+0.5×SAM几何loss；训练使用首步随机恢复 | r1/final.pt |
| 9 | backend/restorer导出 | R1 + D5a | CPU strict FP32 tensor导出 | models/backend.pt、models/restorer.pt |
| 10 | N1缓存 | 本轮backend/restorer；只局部affinity | 32训练源；native1024/overlap.25，显式零噪D5a，原尺寸FP32概率融合 | n1_cache/manifest.json与逐源NPZ |
| 11 | NAF | 随机G/D；冻结真实N1缓存 | G width48；1000updates，B16，256crop/stride50/D4；FP32；G lr1e-3/D1e-4，53000步cosine日程前1000步；GAN热身64后权重.01；容差2；skeleton+dilate8 | naf_train/latest.pt |
| 12 | NAF导出 | 本轮NAF G | strict加载/重载，移除D/optimizer/RNG | models/naf.pt |
| 13 | 汇总 | 本轮三模型 | CPU有限state与总参数上界检查，保存文件身份 | training_export.json |

完整参数位于8份展开配方：stage1、joint、semantic、affinity、d5a、r1、naf、inference。inference配方为N1缓存提供融合算术，其数据与模型路径由本轮产物替换。训练算法显式调用training/source内各成熟trainer；根train.py只负责新编排，Stage1脚本为training/source/train.py。

执行：`python train.py --config training/config.yaml --output-dir outputs/new_training`。查看顺序：加`--plan`；检查实际数据与环境：加`--check`。训练结束保留在新run；随后可运行`python build_engines.py --model-dir outputs/new_training/models --output-dir ../AIC-2026-59388388-模型文件/engines_new`。
