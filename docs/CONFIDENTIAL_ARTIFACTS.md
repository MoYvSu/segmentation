# 赛题数据与派生产物保密

用户明确要求：赛题原图、标注、预测及可还原/展示数据的派生内容不得通过代码仓库传播。
所有分支适用，私有GitHub仓库也不构成放行理由；比赛正式提交按用户授权另行处理。

## 每个克隆的防护

```bash
python tools/confidential_artifacts_guard.py --install
python tools/confidential_artifacts_guard.py --staged
```

安装器将检查器复制到本克隆的Git公共目录，安装独立的pre-commit/pre-push hooks，
并补充本地`info/exclude`。这些副本跨分支生效；新克隆和服务器克隆必须分别安装。
已有自定义hooks或`core.hooksPath`时拒绝覆盖，应显式整合，不得直接禁用。

- pre-commit检查完整暂存树，拒绝产物目录、图片/数组/权重/压缩包、逐图类别文件，
  并检查常见改名图片及内嵌base64原图。
- pre-push检查实际待推送引用的全部可达历史；即使最新版本已删除，旧产物仍会被拒绝。
  只允许分支和标签引用，禁止将`refs/codex/*`等内部快照或备份引用推送出去。
- `.gitignore`防误暂存，Git hooks防常见误提交/误推送；两者不能代替内容审阅，
  也不能阻止浏览器上传、其他克隆或被人为绕过的操作。禁止`--no-verify`和对产物`git add -f`。

`output/`与`outputs/`全部本地保留，不以文件后缀区分其中哪些产物可以公开。
需要公开的脚本应移到`tools/`等代码目录，汇总文字经检查后放入`docs/`；不得顺带搬入原图、掩码、
逐图输出JSON或带内嵌图像的笔记本。代码推送授权不等于数据发布授权。

## 已提交或已推送时

1. 停止传播，在仓库之外保留私有恢复副本；保护未提交代码和本地数据。
2. 核查远程全部分支、标签、PR引用和历史；仅从工作树删除或`git rm --cached`不等于清除历史。
3. 在隔离副本用`git-filter-repo`清理确定的产物路径，检查其他路径是否存在相同内容。
4. 检查非产物代码未改变。逐个列明待替换远程引用及旧、新SHA，用明确的lease保护避免覆盖并发更新。
   不使用`git push --mirror`，不把私有备份和内部引用上传。
5. 清理旧克隆后再恢复推送；不要合并旧历史。PR只读引用及缓存可能需要GitHub Support处理，
   无法承诺收回其他人的下载或克隆，也不能以远程分支清理成功冒充所有副本已消失。

处理边界参考：[GitHub官方敏感数据移除说明](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)。
