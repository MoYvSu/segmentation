# CLAUDE.md

Claude Code 的仓库入口。仓库级代理规则由下方导入的 `AGENTS.md` 提供；本文件只补充 Claude Code
特有的读取顺序、本机环境事实与操作约定，不在这里重复项目介绍、当前成绩、命令和模块清单。

@AGENTS.md

赛题保密是最高级项目硬约束：`output/`与`outputs/`及全部数据派生产物不得提交或推送，
不能以实验归档为由例外。所有分支提交前遵守`AGENTS.md`中的独立Git防护安装与检查要求。

## 继续读什么

按任务范围选择，不要一次性全读：

- `README.md` — 当前最佳、进度、环境与推理入口。
- `docs/PIPELINE.md` — 当前管线、产物契约、目录职责、服务器保留策略。
- `docs/EXPERIMENT_INDEX.md` — 实验关系、复现入口与文件/版本边界。
- `config/README.md` — 配置目录逐项说明与 `_base` 继承机制。
- `COMPETITION_RULES.md` — 赛题合规边界（评分口径、`uint16`/65535、测试集禁用）。
- `docs/COLOR_SEPARABILITY.md` — 讨论语义阈值、颜色辅助或数据增强前必须先读。

具体实验只读对应的 `docs/*_2026*.md` 专题文档。`AGENTS.md` 已写明：文档、配置与代码冲突时，
先核对实际代码与产物，不得静默选择其中一个。

## 本机环境事实

- 平台 Windows 11 + Git Bash；默认解释器是 Anaconda base（Python 3.13.5），已装
  torch 2.8.0+cu128、pytest 8.3.4，足以跑单元测试。训练环境按 `AGENTS.md` 使用 `sam2_env`。
- `.gitignore` 排除了 `data/raw`、`data/test`、`data/unlabeled`、`data/purified_gt`、`weights/`、
  `output/`、`outputs/`、`segment-anything-2/`。本机通常没有完整数据与权重，缺文件是预期状态，不要试图
  从网络补齐或改动这些忽略规则。

## 测试

```bash
python -m pytest -q --ignore=output --ignore=outputs
```

**全量检查从仓库根目录跑。** 根目录下的 `test_*.py` 也是测试，只跑 `tests/` 会漏掉它们。
`output/`、`outputs/` 中可能存有历史源码副本；Git忽略规则不影响pytest收集，必须显式排除，
避免同名测试冲突和旧模块遮蔽。普通修改按风险选择相关测试，不固定全库数量和耗时。

2026-09-19伙伴全库审计仍报告下列既有失败，本轮语义对齐任务没有扩展修复它：

- `test_training_control.py::Stage0TrainingControlTest::test_validation_reports_boundary_haze_metrics`

此前`test_replace_reference_semantic_uses_challenger_only`的测试桩缺失已修复，不再作为已知失败列出。

修它们属于独立任务，先与用户确认再动。不要用 `-x` 跑全量来推断整体健康度。

## 项目 Skill

仓库自带两个 Skill，规范正文在 `.agents/skills/`；`.claude/skills/` 下只是指向正文的薄指针，
执行前必须先完整读取正文文件。

- `maintain-project-knowledge` — 可复用的实验教训、环境问题、代码库特性写入其 `references/`，
  按 observation → validated → skill-rule → `AGENTS.md` 的生命周期晋升。
- `stop-that-shit` — 范围控制：只做被请求的工作和必要后果。

不要另建知识库，也不要把单次事件直接写进 `AGENTS.md`。

## 约定

- 注释、文档、实验记录与提交信息用中文；提交信息 `<type>: 描述`。
- 新分支用 `codex/` 前缀；每个实验保持可独立回退。
- Skill仍为advisory，没有Skill Guard或Claude Plugin；保密约束另有独立Git pre-commit/pre-push
  hooks。不要往`.claude/settings.json`加hooks，也不要混淆两类机制。
