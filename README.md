# wheeled-legged_RL

面向轮腿机器人的强化学习训练项目，基于 **Isaac Sim + Isaac Lab + RSL-RL**，提供机器人建模、任务训练、策略回放、系统辨识、评测与模型导出工具。

支持 Wheelbipe V14、Wheel-leg V1–V4 和 Deformable 主动悬挂任务。项目按源码、资源、运行入口、工程工具、测试和文档分层组织，三个核心包统一从仓库根目录安装。

## 功能与架构

| 模块 | 内容 | 源码位置 |
| --- | --- | --- |
| 世界与机器人 | 机器人配置、执行器、电机与气簧模型、地形生成 | [`agent_world`](src/agent_world/) |
| 环境与任务 | 任务注册、观测与动作、奖励、状态机、课程和任务合同 | [`agent_tasks`](src/agent_tasks/) |
| 强化学习 | PPO、DreamWaQ、HIMLoco、NP3O，以及 MLP / Transformer 网络与 runner | [`agent_rl`](src/agent_rl/) |
| 运行入口 | 训练、回放、环境查看、策略导出和 RMUC 场地启动 | [`scripts`](scripts/) |
| 工程工具 | 参数辨识、行为评测、实验编排、资源转换和仓库检查 | [`tools`](tools/README.md) |

任务配置选择相应的机器人与算法；机器人描述引用 `assets/` 中的资源。部署所需的观测、动作、历史帧和控制时序由具体任务与模型配置决定。

## 项目结构

```text
wheeled-legged_RL/
├── src/
│   ├── agent_world/           # 机器人配置、执行器、地形生成器
│   ├── agent_tasks/           # 环境、任务、奖励、配置与合同
│   └── agent_rl/              # 算法、网络、runner、存储与环境接口
├── assets/
│   ├── robots/                # 自有 USD、URDF、网格与机械配置
│   └── terrains/              # 场地资源说明；大型地图单独安装
├── scripts/
│   ├── rsl_rl/                # train.py、play.py、export_onnx.py 等入口
│   ├── setup/ubuntu/          # Ubuntu 服务器安装与检查脚本
│   └── utils/                 # 回放配置、轨迹记录与可视化辅助模块
├── tools/                     # 辨识、评测、诊断、实验编排与转换
├── tests/                     # CPU 回归测试
├── docs/                      # 使用指南、任务设计与机械资料
├── models/                    # 本地模型，仅跟踪 README
├── references/                # 本地参考项目，仅跟踪 README
├── .github/workflows/         # 仓库检查与 CPU 测试
├── pyproject.toml             # 统一安装、依赖分组与测试配置
├── Makefile                   # install / check / test
├── play_rmuc.sh               # RMUC 场地回放入口
└── run_gui.sh                 # Arch 桌面图形兼容入口
```

目录职责、旧路径与新路径的对应关系见[项目架构与迁移表](docs/development/architecture.md)。

## 环境与安装

### 环境要求

项目安装要求 Python 3.10–3.11；当前本地训练环境如下。仿真器、PyTorch 和 CUDA 由 Isaac Lab 环境提供，根目录安装不会自动安装这些组件。

| 组件 | 当前本地环境 |
| --- | --- |
| Python | 3.11 |
| Isaac Sim | 5.1.0.0 |
| Isaac Lab | 2.3.2.post1 |
| PyTorch / torchvision | 2.10.0+cu128 / 0.25.0+cu128 |
| RSL-RL | 3.0.1 |

服务器环境配置见 [Ubuntu 安装指南](scripts/setup/ubuntu/README.md)。不同任务的历史环境与复现说明见[实验复现](docs/guides/reproduce.md)。

### 获取与安装项目

在已配置好的 Isaac Lab 环境中执行：

```bash
git clone https://github.com/ZGZ713912/isaac_lab_tasks.git wheeled-legged_RL
cd wheeled-legged_RL
git lfs pull

conda activate isaaclab
python -m pip install --no-deps -e .
```

需要补齐训练工具或开发依赖时：

```bash
python -m pip install -e '.[train,dev]'
```

`agent_world`、`agent_tasks`、`agent_rl` 保留原导入名，无需分别安装子目录。`make install` 等价于根目录的可编辑安装。

### 资源与模型

机器人资源保存在 `assets/robots/`，USD、URDF 与网格的相对依赖树需完整保留。部分文件使用 Git LFS；地图包和预训练模型需单独准备。

| 内容 | 位置 | 获取说明 |
| --- | --- | --- |
| 自有机器人描述 | `assets/robots/` | 随仓库跟踪，LFS 文件使用 `git lfs pull` |
| RMUC2026 场地 | `assets/terrains/RMUC2026_V2_gray_fence_revision_03/` | 按[场地说明](assets/terrains/README.md)安装完整资源包 |
| 本地模型与历史预训练权重 | `models/`、`models/pretrained/` | 按[模型管理](models/README.md)保存权重与配套参数 |

资源默认从当前 checkout 的 `assets/` 解析；使用 wheel 或外置资源目录时，可设置 `WHEELED_LEGGED_ASSETS_DIR=/absolute/path/to/assets`。部分旧 Wheelbipe 配置还需额外的历史资源包，详见[资源说明](assets/README.md)。

## 快速使用

以下命令从仓库根目录执行，使用已激活的 Isaac Lab Python 环境。

### 查看任务

```bash
python scripts/list_envs.py
```

该入口会以无窗口模式启动 Isaac Sim 并列出已注册任务。代表性任务如下，完整变体以源码注册和脚本输出为准。

| 系列 | 代表性任务 ID | 配置位置 |
| --- | --- | --- |
| Wheelbipe V14 | `Robotics-Wheelbipe-V14-Flat-v0` | [`wheelbipe_V14`](src/agent_tasks/direct/wheelbipe/wheelbipe_V14/) |
| Wheel-leg V2 | `Robotics-Wheel-Leg-V2-Flat-v0` | [`wheel_leg_v2`](src/agent_tasks/direct/wheel_leg_v2/) |
| Wheel-leg V4 | `Robotics-Wheel-Leg-V4-Flat-v0` | [`wheel_leg_v4`](src/agent_tasks/direct/wheel_leg_v4/) |
| Deformable Guarded MLP | `Robotics-Deformable-Suspension-Guarded-History-MLP-Real2Sim-v4` | [`deformable_suspension`](src/agent_tasks/direct/deformable_suspension/) |

### 训练与回放

以 Wheel-leg V4 平地任务为例：

```bash
python scripts/rsl_rl/train.py \
  --task Robotics-Wheel-Leg-V4-Flat-v0 \
  --num_envs 4096 --max_iterations 20000 --device cuda:0 --headless
```

训练日志和 checkpoint 默认保存在 `logs/rsl_rl/`。替换为实际模型路径后回放：

```bash
CHECKPOINT="/absolute/path/to/run/model_XXXX.pt"
python scripts/rsl_rl/play.py \
  --task Robotics-Wheel-Leg-V4-Flat-Play-v0 \
  --num_envs 1 --checkpoint "$CHECKPOINT" --keyboard
```

Deformable 的辨识、恢复训练和 Guarded 训练使用对应的专用流程，见[训练记录](docs/deformable-real2sim-training.md)与 [Guarded 训练指南](docs/deformable-guarded-training.md)。

### 策略导出

```bash
python scripts/rsl_rl/export_onnx.py \
  --checkpoint "$CHECKPOINT" \
  --output_dir models/wheel_leg_v4/exported
```

导出时保留模型对应的 `params/agent.yaml`、`params/env.yaml` 以及需要的 Real2Sim 参数。模型的任务、观测/动作合同、网络结构和历史时序必须匹配，文件位置不能替代合同校验。

### RMUC 场地回放

安装完整地图并准备对应模型后，使用统一入口。指定当前激活环境的解释器，先预检启动命令：

```bash
export ISAACLAB_PYTHON="$(command -v python)"
./play_rmuc.sh wheel-leg --checkpoint "$CHECKPOINT" --dry-run
# 预检后启动回放
./play_rmuc.sh wheel-leg --checkpoint "$CHECKPOINT"
```

`deformable` 车型使用 `./play_rmuc.sh deformable --checkpoint /absolute/path/to/deformable/run/model_XXXX.pt`；ONNX 模型使用 `--onnx PATH`，并按需通过 `--policy-config PATH` 指定配套配置。省略模型路径时，入口使用本机固定的历史 checkpoint，文件需预先存在。

模型选择、控制按键、相机、场地碰撞覆盖及默认 Wheel-leg 模型的站立训练范围，见 [RMUC 回放指南](docs/rmuc-field-play.md)。Arch 桌面需要现有图形兼容环境时，可使用 `./run_gui.sh python <script> ...`。

## 开发与验证

```bash
# 文件清单、提交边界、Python / Shell 语法
make check

# CPU 回归测试
make test

# 构建 Python wheel
python -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
```

`make` 默认使用当前 `python`，可用 `make test PYTHON=/absolute/path/to/python` 指定环境。独立 CPU 测试依赖见 [`requirements/test.txt`](requirements/test.txt)，CI 配置见 [GitHub Actions](.github/workflows/tests.yml)。测试只收集 `tests/`；GPU 训练、完整仿真回放和实车验证分别执行。

## 版本管理约定

- 跟踪源码、任务配置与合同、自有机器人资源、测试和文档。
- `参考例程/` 与 `references/` 中的外部项目保留在本地，由 `.gitignore` 排除；`references/README.md` 作为使用说明保留。
- `models/` 仅跟踪说明文件；权重、TensorBoard 日志、视频、运行快照和评测产物保存在被忽略的目录中。
- `logs/`、`outputs/`、`runs/`、`reports/`、大型 RMUC 地图与本机状态不进入提交。

本地参考例程不参与项目安装或运行。取消跟踪只影响当前及后续提交，已有 Git 历史保留。具体规则见 [`.gitignore`](.gitignore)与[贡献指南](CONTRIBUTING.md)。

## 文档

- [文档索引](docs/README.md)：任务、回放、辨识、设计与历史验证记录。
- [完整使用说明](docs/guides/usage.md)与[实验复现](docs/guides/reproduce.md)。
- [项目架构与迁移表](docs/development/architecture.md)。
- [Wheel-leg V2 训练](docs/wheel_leg_v2_training.md)。
- [Deformable 运行时保护](docs/deformable-runtime-guard.md)与[第三人称场地回放](docs/deformable-rmuc-play.md)。

## 许可与归属

项目根目录许可见 [LICENSE](LICENSE)。部分算法派生自第三方项目，各自的许可、版权与适用文件见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
