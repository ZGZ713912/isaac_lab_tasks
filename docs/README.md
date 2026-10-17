# 文档索引

## 开始使用与开发

- [项目首页](../README.md)
- [完整使用说明](guides/usage.md)：任务、训练、回放、算法和轨迹记录
- [实验复现](guides/reproduce.md)：历史环境与预训练模型复现
- [项目架构与迁移表](development/architecture.md)
- [贡献指南](../CONTRIBUTING.md)
- [Ubuntu 服务器配置](../scripts/setup/ubuntu/README.md)

## 场地回放

- [统一 RMUC 回放入口](rmuc-field-play.md)
- [Deformable 场地回放](deformable-rmuc-play.md)

## Deformable

- [Real2Sim 设计](deformable-real2sim.md)、[拟合与回放](deformable-real2sim-fitted.md)、[训练与验证记录](deformable-real2sim-training.md)
- [动态悬挂](deformable-dynamic-v1.md)、[关节水平控制](deformable-joint-leveling.md)、[水平控制短训](deformable-leveling-short-train.md)
- [Minangle 约束](deformable-minangle-v2.md)、[精度流程](deformable-precision-v2.md)、[低滑移任务](deformable-low-slip-v2.md)
- [五帧 MLP](deformable-mlp5.md)、[MLP 恢复与评测记录](deformable-mlp5-recovery.md)
- [Guarded 训练](deformable-guarded-training.md)、[运行时保护](deformable-runtime-guard.md)
- [主动悬挂设计计划](design/deformable-active-suspension-plan.md)

## Wheel-leg 与机械资料

- [V1 架构](wheel_leg_v1_architecture.md)、[车轮碰撞](wheel_leg_v1_wheel_collider.md)
- [V2 训练](wheel_leg_v2_training.md)
- [串联腿运动学](mechanics/Wheel_leg_V1串联正运动学解算.md)
- [串联腿动力学资料](mechanics/serial-leg-dynamics.pdf)
- [气簧参数](mechanics/gas-spring-parameters.jpg)

历史验证文档中的结果只对应其具体任务、配置、模型和当次环境。
