在生成git commit message时使用变更描述使用中文，可以参考以下规范：
规范的 commit message:
<type>(<scope>): <description>

type为前缀，也就是提交类型
scope是可选的，就是本次涉及到的页或者模块
description就算变更描述。

当前比较流行的前缀：
feat 用于新增功能（feature）
fix 用于修复 bug
docs 用于文档的变更（documentation）
style 代码风格的修改（不影响功能，比如空格、格式化、缺少分号等修正）
refactor 用于代码重构（既不修复 bug，也不增加功能）
test 用于添加或修改测试用例
chore 用于构建过程或辅助工具的变动（不影响源代码或测试用例）
perf 用于提升性能的代码改动
build 用于影响构建系统或外部依赖的更改（例如：gulp、webpack、npm）
revert 用于回滚之前的提交
ci 用于修改 CI 配置文件和脚本（如 GitHub Actions、Jenkins）
deps 用于依赖的变更（更新、删除或新增）

