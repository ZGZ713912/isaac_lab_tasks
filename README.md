# wheeled-legged_RL

基于 Isaac Sim、Isaac Lab 和 RSL-RL 的轮腿机器人训练项目，包含 Wheelbipe、Wheel-leg V1–V4 和 Deformable 主动悬挂任务，以及训练、回放、辨识、评测和策略导出工具。

## 项目结构

```text
.
├── src/
│   ├── agent_world/      # 机器人配置、执行器、地形生成器
│   ├── agent_tasks/      # 环境、任务注册、奖励、观测和任务配置
│   └── agent_rl/         # PPO 变体、网络、runner、存储和环境接口
├── assets/
│   ├── robots/           # 自有 USD、URDF、网格及机械配置
│   └── terrains/         # 场地资源说明；大型地图单独安装
├── scripts/              # 训练、回放、导出入口和环境安装脚本
├── tools/                # 辨识、实验编排、诊断和资源转换
├── tests/                # 无需启动仿真器的回归测试
├── docs/                 # 使用说明、任务设计、机械资料和开发指南
├── models/               # 本地模型及预训练权重，只提交说明文件
├── references/           # 本地参考仓库和下载包，只提交说明文件
├── .github/workflows/    # 仓库检查和 CPU 回归测试
├── pyproject.toml        # 统一安装、依赖分组和测试配置
└── Makefile              # install / check / test
```

`logs/`、`outputs/`、`runs/`、`reports/`、`参考例程/` 和模型文件由 Git 忽略。源码不依赖本地参考例程；目录职责和迁移映射见 [项目架构](docs/development/architecture.md)。

## 安装

在已经配置好 Isaac Sim / Isaac Lab 的 Python 环境中执行。本项目保留原有 `agent_world`、`agent_tasks`、`agent_rl` 导入名，统一从根目录安装：

```bash
conda activate isaaclab
python -m pip install --no-deps -e .
# 补齐训练工具依赖和开发工具时使用：
python -m pip install -e '.[train,dev]'
```

Python 要求为 3.10–3.11；现有训练环境使用 Python 3.11、Isaac Sim 5.1、Isaac Lab 2.3.2 和 RSL-RL 3.0.1。仿真器、PyTorch 和 CUDA 由外部环境管理。[Ubuntu 安装脚本](scripts/setup/ubuntu/README.md) 提供现有服务器部署流程。机器人数据保留完整相对依赖树；LFS 资源需执行 `git lfs pull`。

## 运行

以下命令在仓库根目录执行。模型路径需替换为本地实际文件。

```bash
# 列出任务
python scripts/list_envs.py

# 训练
python scripts/rsl_rl/train.py --task=<task_name> --num_envs=4096 --headless

# 回放
python scripts/rsl_rl/play.py --task=<task_name> --num_envs=1 --checkpoint=<model_path> --keyboard

# 在本地安装的 RMUC 场地上回放
./play_rmuc.sh deformable
./play_rmuc.sh wheel-leg

# 仓库检查和 CPU 回归测试
make check
make test
```

Arch 桌面需要现有图形兼容环境时，可使用 `./run_gui.sh python <script> ...`。场地和 checkpoint 的安装位置、任务匹配及操作说明见 [RMUC 回放指南](docs/rmuc-field-play.md)。

## 文档与协作

- [文档索引](docs/README.md)：任务训练、回放、辨识及历史验证记录。
- [完整使用说明](docs/guides/usage.md)与[实验复现](docs/guides/reproduce.md)。
- [目录迁移和开发约定](docs/development/architecture.md)。
- [贡献指南](CONTRIBUTING.md)：安装、测试、资源和提交边界。
- [本地参考例程](references/README.md)与[模型管理](models/README.md)。

本项目包含多种第三方算法，许可及归属见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [LICENSE](LICENSE)。
