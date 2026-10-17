# 项目结构与迁移

## 分层

本仓库采用 RMCS 的工作空间分层思路和 isaac_wheeled_rl_train 的源码、工具、测试分离方式，使用根目录 `pyproject.toml` 统一安装。三个原有包名与任务 ID 保持兼容。

| 位置 | 职责 | 版本管理 |
| --- | --- | --- |
| `src/agent_world/` | 机器人配置、执行器、地形生成 | 跟踪 |
| `src/agent_tasks/` | 环境、观测/动作、奖励、状态机、任务配置与合同 | 跟踪 |
| `src/agent_rl/` | PPO、网络、runner、存储、向量环境接口 | 跟踪 |
| `assets/robots/` | 自有 USD、URDF、网格与机械配置 | 跟踪，部分文件沿用 LFS |
| `assets/terrains/` | 地图说明及独立安装的场地包 | 说明跟踪，大型地图忽略 |
| `scripts/` | 训练、回放、导出、查看环境入口 | 跟踪 |
| `scripts/setup/ubuntu/` | 既有服务器环境安装及检查 | 跟踪 |
| `scripts/utils/` | 现有入口和任务共享的回放/轨迹辅助模块 | 随项目安装，保留兼容导入 |
| `tools/` | 实验编排、辨识、诊断与资源转换 | 跟踪 |
| `tests/` | CPU 回归测试 | 跟踪 |
| `docs/` | 指南、任务设计和机械资料 | 跟踪 |
| `models/`、`references/` | 本地模型和外部参考代码 | 只跟踪说明文件 |
| `logs/`、`outputs/`、`runs/`、`reports/` | 运行产物与过程证据 | 忽略 |
| `.local/` | 工作站兼容库、旧安装元数据和本机状态 | 忽略 |

调用方向以入口 → 任务/算法 → 世界配置 → 物理资源为主。任务配置与部署合同仍位于所属任务中，避免拆散它们和运行代码的版本关系。`scripts.utils` 的现有导入暂时保留并纳入统一安装，直接运行入口也会优先加载当前 checkout 的 `src/`。

## 旧路径迁移表

| 旧位置 | 新位置 |
| --- | --- |
| `source/agent_world/agent_world/` | `src/agent_world/` |
| `source/agent_tasks/agent_tasks/` | `src/agent_tasks/` |
| `source/agent_rl/agent_rl/` | `src/agent_rl/` |
| `source/agent_world/agent_world/assets/usd_files/` | `assets/robots/` |
| `scripts/tools/test_*.py` | `tests/test_*.py` |
| `scripts/tools/` 中的工具 | `tools/` |
| `source/usd2urdf_tools/usd2urdf.py` | `tools/usd2urdf/usd2urdf.py` |
| `source/docs/` | `docs/mechanics/` |
| `source/agents/` | `docs/design/` |
| `ubuntu_setup/` | `scripts/setup/ubuntu/` |
| `REPRODUCE.md` | `docs/guides/reproduce.md` |
| 原 README 中的详细用法 | `docs/guides/usage.md` |
| `气簧参数.jpg` | `docs/mechanics/gas-spring-parameters.jpg` |
| `pretrained/` | 本地 `models/pretrained/` |
| `runs_v5/` | 本地 `outputs/legacy/runs_v5/` |
| `source/usd2urdf_tools/out_wheelbipe/` | 本地 `outputs/legacy/usd2urdf/out_wheelbipe/` |
| `debug/` | 本地 `logs/legacy/` |
| 根目录参考下载压缩包 | 本地 `references/` |
| `参考例程/` | 本地原位置保留，取消 Git 跟踪 |

运行日志、模型和旧快照保留原字节；其中的历史路径和哈希记录不回写。新运行的源码快照使用新目录。旧日志中的绝对路径需按上表定位，不应把历史运行快照作为当前可执行源码。

## 资源与安装

USD 和 URDF 的内部目录整体迁移，内部相对引用保持不变。`agent_world.RobotAssetPath` 指向 `assets/robots/`，`TerrainAssetPath` 指向 `assets/terrains/`。外置资源通过 `WHEELED_LEGGED_ASSETS_DIR` 指定。冻结的 Wheel-leg V2 合同仍保留 `usd_files/...` 字段，加载器将该前缀解析到新机器人目录，合同内容与摘要不变。

根目录可编辑安装会发现三个包的全部子模块，包括此前被通用 `env/` 规则漏掉的 `agent_rl.rsl_rl.env`。JSON 配置和合同作为 package data 纳入安装；机器人网格、USD、地图和权重不打入 Python wheel，需要保留资源 checkout 或使用环境变量指定。

## 检查

```bash
make check
make test
python -m pip wheel --no-deps --no-build-isolation --wheel-dir /tmp/wheeled-wheels .
```

GitHub Actions 执行文件清单/语法检查和 CPU 回归。GPU 训练、完整仿真回放、物理控制和实车验证分别执行；目录调整不授予这些能力结论。
