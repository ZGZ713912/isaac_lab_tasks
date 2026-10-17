# 项目资源

`robots/` 存放本项目自有机器人描述，包含 USD 分层文件、URDF、机械网格和配置；`terrains/` 存放场地安装说明及本地资源。

原 `source/agent_world/agent_world/assets/usd_files/` 整体迁移为 `assets/robots/`，资源内部相对路径与文件内容保持不变。Python 机器人配置留在 `src/agent_world/assets/`。

`agent_world.RobotAssetPath` 和 `agent_world.TerrainAssetPath` 分别提供机器人和场地绝对路径。默认指向当前 checkout 的 `assets/`；使用 wheel 安装时，通过 `WHEELED_LEGGED_ASSETS_DIR=/absolute/path/to/assets` 指定资源目录。

部分旧 Wheelbipe 配置需要额外的历史资源包，源码目录迁移不会补齐原本缺失的模型。转换工具在 `tools/` 和 `tools/usd2urdf/`；临时转换输出应写入 `outputs/`。
