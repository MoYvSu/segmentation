# 复赛提交压缩包格式

复赛提交的是一个压缩包，压缩包内必须同时包含**评测结果**和**代码/模型**两部分。评测程序只读取 `results/`，不会执行选手提交的代码，也不会从 `code_model/` 读取预测结果。

## 推荐目录结构

```text
submission.zip
├── results/
│   ├── test_069_inst.png
│   ├── test_069_class.json
│   ├── test_070_inst.png
│   ├── test_070_class.json
│   └── ...
└── code_model/
    ├── model/
    ├── infer.py
    ├── requirements.txt
    └── README.md
```

## 结果文件要求

- `results/` 下每张测试图片必须有一对文件：`test_XXX_inst.png` 和 `test_XXX_class.json`。
- 复赛当前编号为 `test_069` 至 `test_168`，共 100 张；文件名中的三位编号不能省略。
- PNG 必须是 **uint16 的单通道 PNG**，尺寸必须与对应原图一致；实例 ID 使用正整数，背景为 0。不要提交 RGB、RGBA 或其他多通道 PNG。
- JSON 顶层必须是对象，例如 `{"1": 0, "2": 1}`；类别值只能使用 `0`（珠光体）或 `1`（铁素体）。
- `results/` 目录不要再放同名文件的多个副本，避免评测程序无法判断应读取哪一份。

## 代码和模型要求

- 所有模型、推理代码和运行依赖请放在 `code_model/`，不要与 `results/` 混放。
- 评测阶段当前只按 `results/` 中的文件计算分数，不会自动运行 `code_model/` 中的程序。
- 请在 `code_model/README.md` 中写明模型文件、入口脚本、依赖版本和推理命令，便于复赛复核。
- 不要提交密码、Token、数据库连接信息或其他无关的大文件。

## 压缩和兼容说明

- 压缩包请命名为 `初赛结果_团队名.zip` 或 `复赛结果_团队名.zip`，例如 `复赛结果_星火队.zip`。
- 请压缩整个 `submission/` 目录后提交，确保解压后能看到 `results/` 和 `code_model/`。
- 也接受外面再套一层队伍目录，例如 `my_team/results/...`；评测程序会自动查找结果目录。
- 旧格式“直接把 PNG/JSON 放在压缩包根目录”仍可兼容，但复赛请使用上述分目录格式。
- `result/`、`predictions/`、`output/`、`outputs/` 等目录名也可被识别，但推荐统一使用 `results/`。
- 压缩包内不得使用 `../` 等越界路径；包含此类路径的压缩包会被判为无效提交。

## 提交前自检

1. 解压压缩包，确认存在 `results/` 和 `code_model/`。
2. 确认 `results/` 中有 100 对 `test_069` 至 `test_168` 文件。
3. 随机打开 PNG，确认是单通道且尺寸正确。
4. 随机检查 JSON，确认实例 ID 与 PNG 中的实例 ID 完全一致。