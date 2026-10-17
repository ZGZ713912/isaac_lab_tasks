# 开发与提交

## 安装和检查

使用已配置好的 Isaac Lab Python 环境，在根目录执行：

```bash
python -m pip install --no-deps -e .
python -m pip install -e '.[dev]'
make check
make test
```

`make check` 检查 Git 文件清单、Python 语法、Shell 语法和必要项目文件；`make test` 只收集 `tests/`。CPU 回归通过不代表 Isaac Sim 场景、实车或训练效果已验证。

## 文件放置

- 可复用实现放在 `src/` 三个包中，保持原有导入名。
- 运行入口放在 `scripts/`；辨识、诊断、实验编排和转换工具放在 `tools/`。
- 测试放在 `tests/`，模拟器测试应另行显式执行。
- 自有机器人资源放在 `assets/robots/`，保留 USD、URDF 与网格的相对路径。
- 配置和部署合同随所属任务存放；更改目录时保留冻结合同的内容及策略 ABI。
- 使用与设计说明放在 `docs/`；新增入口同时补充文档索引。

## Git 边界

参考仓库放在 `references/` 或本地 `参考例程/`；权重放在 `models/`；日志、数据集、视频、转换产物和运行快照放在 `logs/`、`outputs/`、`runs/` 或 `reports/`。上述内容不进入版本库，目录说明文件除外。不要使用 `git add -f` 绕过此约定。

提交前检查 `git diff --cached --stat` 和 `git status --short`。必要源码目录（例如 `src/agent_rl/rsl_rl/env/`）必须被跟踪，不能用通用 `env/` 忽略规则屏蔽。
