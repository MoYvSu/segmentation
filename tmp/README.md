# 历史辅助材料

本目录保留实验期间使用过的传输、环境准备、诊断与文档生成材料。正式训练和分析入口见
[实验索引](../docs/EXPERIMENT_INDEX.md)。

- 根目录的 `audit_mask_set_remote.py`、`setup_mask_set_remote.py`、`launch_mask_set_remote.py`
  和 `package_mask_set.py` 属于首轮掩码模型的部署现场，运行前需要核对当时服务器路径。
- `20260910_diagnostic_remote_source/`、`remote_monitor_sync_check_20260907/` 保存当时取回的
  源码或配置，用于版本比较；不作为当前模块导入路径。
- `pdfs/` 保存赛题摘要与技术报告的生成脚本；`musam_oracle_smoke/` 保存诊断小验证报告。
- 其余目录中的压缩包、派生图和缓存仍在本地。忽略规则只控制版本库收录，不删除内容。

[.gitignore](.gitignore) 保留脚本、说明、配置和轻量指标，排除打包与渲染产物。
