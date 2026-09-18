# 新闭链模型交付及后续验证

完整压缩包：`chassis_closedchain_20260917.zip`（约 5.9 MiB）。
完整文件夹：`urdf/`；Isaac 入口：`urdf/robot.usda`。

- 15 个刚体，14 个树转动关节轴，4 个闭合球约束。
- 每个刚体含质量、质心、惯量、可视网格和碰撞几何，总质量 12.752 kg。
- 标准 URDF 单独导入只有树，需要同时处理 `constraints.json`；USD 已包含闭合约束。
- 拆分杆件惯量是均匀密度研究估计，实际驱动映射待机械参数确认。
- ZIP CRC 及逐文件一致性检查通过。

ZIP SHA256：`ceaf7df719b1c4e366400ca3799da5f294e2c25498c53ff54542557154b111d3`

打包后新增 GPU PhysX 验证：真实求解器读到全部 15 刚体、14 个关节轴；重力和小力矩
最大闭合误差 0.7261 mm，关闭约束对照误差 110.9685 mm；无逐帧写被动杆姿态。
独立单环境 PPO 完成 16 次更新，actor/critic 参数实际变化且有限，动作下最大闭合误差 0.4532 mm。
这只是小规模工程试训，不是站立效果验收。

包内 `isaac_validation=not_run` 保留生成时状态；上述后续验证关联的是
manifest SHA256 `715f8e5bf8259543aacdb4d29fc97962f37a943ea1157d22a09ef4bbdff92016`，
未覆盖原模型包。完整记录见工作区 `isaac_wheeled_rl_train-60/docs/CHASSIS_CLOSEDCHAIN_20260917.md`。
