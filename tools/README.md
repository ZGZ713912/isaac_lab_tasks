# 工程工具

本目录存放辨识、评测、实验编排和资源转换工具。训练与交互入口位于 `scripts/`，回归测试位于 `tests/`。

| 工具 | 用途 |
| --- | --- |
| `check_repository.py` | 检查提交文件边界和源码语法，无需仿真依赖 |
| `deformable_suspension_eval.py` | 固定条件下评估悬挂策略 |
| `deformable_feasibility.py` | CPU 几何可行性检查 |
| `deformable_real2sim_identify.py`、`deformable_*_fit.py` | 记录读取、参数辨识与模型拟合 |
| `deformable_*_train.py`、`run_deformable_guarded_experiment.py` | 训练流程编排与源文件封存 |
| `deformable_*_audit.py`、`deformable_*_report.py` | 审核行为证据与生成报告 |
| `prepare_*_urdf.py`、`convert_*_urdf.sh` | 机器人资源转换 |
| `usd2urdf/usd2urdf.py` | USD → URDF 导出 |

在已配置好的项目环境中，从根目录运行 `python tools/<tool>.py --help` 查看参数。需要仿真器的工具可能在参数解析前初始化 Isaac Sim。输出写入 `outputs/` 或 `logs/`，避免把数据、权重和转换网格放入本目录。
