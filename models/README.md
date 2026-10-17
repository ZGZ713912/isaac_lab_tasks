# 本地模型

原 `pretrained/` 已迁移到 `models/pretrained/`，权重和训练参数保留在本地，不进入 Git。

新模型按任务和运行编号存放，例如 `models/deformable/<run>/`。回放时显式传入 checkpoint / ONNX 路径，并保留配套任务参数、网络结构、观测/动作合同和模型来源。

训练过程的模型和日志仍可直接保存在 `logs/rsl_rl/`，实验评测输出保存在 `outputs/`；两者均被忽略。需要共享模型时使用独立的模型存储或发布附件，源码中记录可复现的来源与合同。
