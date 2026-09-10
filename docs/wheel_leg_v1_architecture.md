# Wheel_leg_V1 训练架构、奖励与地形/摩擦设计说明

- 日期：2026-09-11
- 状态：done（当前代码快照说明；行号会随代码变动漂移，以函数名/类名为准）
- 相关代码：
  - 资产：`source/agent_world/agent_world/assets/wheel_leg_V1.py`
  - 任务包：`source/agent_tasks/agent_tasks/direct/wheel_leg_v1/`
  - 地形库：`source/agent_tasks/agent_tasks/manager/mdp/isaaclab/terrains.py`
  - 算法：`source/agent_rl/agent_rl/rsl_rl/`（PPO / DreamWaQ / HIM / NP3O）
  - 脚本：`scripts/rsl_rl/{train,play}.py`

---

## 0. 两个问题的直接回答

### 0.1 轮子摩擦力等场地设计好了吗？

**仿真训练侧已经设计好并正在生效**，但属于"域随机化 + 通用地面材质"级别，**不是实机轮胎—地面实测标定**。

已配置的内容：

| 项 | 值 | 位置 |
|---|---|---|
| 全局仿真物理材质（Isaac 默认地面材质） | static=1.0 / dynamic=1.0 / restitution=0.0，combine=multiply | `wheel_leg_base/env_cfg.py`（`WheelLegBaseEnvCfg.sim`） |
| Rough 地形材质 | 同上（1.0/1.0/0.0，multiply） | `wheel_leg_task/cfg_utils.py:_apply_v14_rough_runtime_cfg` |
| **轮子接触摩擦（训练随机化，关键项）** | 静摩擦 0.5~1.2，动摩擦 0.4~1.0，恢复 0.02~0.2 | `wheel_leg_task/env_cfg.py:EventCfgV14.physics_material` |
| 车体材质（故意打滑，防作弊） | 静/动摩擦 0.01~0.1，恢复 0.02~0.2 | `EventCfgV14.base_material` |
| 轮关节库仑/粘性摩擦 | static +0.05~0.25，viscous +0.0~0.01 | `EventCfgV14.wheel_joint_frictions` |
| 腿关节摩擦 | static/dynamic +0.25~1.0，viscous +0.05~0.2 | `EventCfgV14.leg_front/rear_joint_frictions` |
| 执行器 PD 增益随机化 | kp/kd ×0.75~1.25 | `EventCfgV14.robot_joint_stiffness_and_damping` |
| 轮电机出力扰动 | 力矩 ×0.9~1.1（噪声/偏置当前为 0） | `EventCfgV14.wheel_effort_noise` |
| 摩擦在线调试 | Play 可打印左右轮实际 PhysX 摩擦系数与接触力峰值 | `wheel_leg_base/env.py:_maybe_print_wheel_material_debug` |

需要注意的缺口/假设：

1. **轮子碰撞体是 STL 网格 → convex hull**（`urdf_V4.0.urdf` 中 `L/R_link3` 用 mesh，转换 `config.yaml: collider_type=convex_hull`），不是解析圆柱；滚动接触是凸包近似。
2. PhysX 摩擦是**各向同性**，没有对轮胎做横向/纵向各向异性或滚动阻力建模；对"两轮差速轮腿"这是合理简化，但不等价于实机橡胶胎。
3. 地形摩擦在 rough 里固定 1.0，没有按地形类型（钢板/木箱/地毯）区分材质。
4. 训练时随机化范围来自经验值（沿用 wheelbipe/25v3），**没有用实机测得的 μ 曲线标定**；上实机前建议做系统辨识再收窄范围。
5. Play 模式的事件表 `EventCfgV14_Play` 与训练版相同，**演示时随机化仍然生效**（只关了部分调试打印），想固定手感需手动把对应 `EventTerm` 置 `None`。

### 0.2 base / task / terrain 三层分别干什么

一句话：
- **base（wheel_leg_base）**：RL 基座。定义环境类本体——场景、机器人、动作下发、观测拼接、**所有奖励公式**、终止/重置、域随机化/噪声/延迟，以及基础配置类。
- **terrain（wheel_leg_terrain）**：地形命令层。把"机器人当前站在哪种地形上"翻译成"速度/角速度/身高指令覆盖 + 特殊模式开关 + 朝向锁定规则"，并管理地形可视化标记。
- **task（wheel_leg_task）**：任务层。负责 gym 注册、具体任务的参数表（含**奖励权重**、地形生成器、课程、算法变体观测空间）、粗糙地形高度课程、速度轨迹录制、简单根复位、越界重置。

一句话记法：**公式在 base、权重和任务在 task、地形翻译在 terrain**。

### 0.3 坐标系与关节符号约定（2026-09-11 修正）

**背景**：SolidWorks 原始导出的 base 坐标系是
`+X=左、+Y=车尾、+Z=上`，轮轴沿 X → 轮子沿 Y 滚动；而整套 RL 代码（奖励/指令/观测）
按 wheelbipe 的 `+X=前进、+Y=左、+Z=上` 约定编写。这导致平地训练出现一系列症状：

- `track_lin_vel_xy` 实际要求机器人"横移"（两轮平衡车物理上做不到）→ 学不会、存活短；
- `no_fork` 把左右轮距（≈0.4 m）当成"劈叉"恒罚（`no_fork=-1`、`no_fork_square` 每步 ≈-3.4）；
- 左右腿关节轴镜像，同号动作 = 一伸一缩 → 落地后两腿不对称。

**修正**（不动任何几何尺寸，只改坐标系/正方向）：
`scripts/tools/prepare_wheel_leg_v1_urdf.py` 以 SolidWorks 原始导出为输入，
做下面两件事后写出 `urdf/urdf_V4.0.urdf`。
坐标轴不对的原始 URDF（`urdf_V4.0_solidworks.urdf`）与对应的旧 USD 已从仓库删除，
避免误用；如需重新规范化，可从 git 历史 `21f2a31` 取回原始文件后再运行脚本：

1. **根坐标系绕 Z 转 +90°**：新 `+X=前进`、`+Y=左`、`+Z=上`；L 腿在 +Y、R 腿在 -Y，轮轴沿 Y。
   （只变换 base_link 的 inertial/visual/collision origin 与父为 base_link 的 L/R_joint1 origin；
   子关节局部坐标不变。）
2. **统一左右腿关节正方向**（翻转 `L_joint2 / R_joint1 / R_joint3` 的 axis 符号）：

| 关节 | 约定 |
|---|---|
| `joint1`（等效髋/大腿） | 增大 = 轮端向前摆；两侧同号 = 对称 |
| `joint2`（等效膝/小腿） | **减小 = 伸腿蹬地（站高增大）**，增大 = 收腿锁腿；两侧同号 = 对称 |
| `joint3`（轮） | 正转 = 轮子向前滚；两侧同号 = 同向前进 |

修正后零位（q=0）站高 ≈ 0.229 m（base 原点离地，轮半径 0.06 m），
身高可达范围约 [0.05, 0.31] m，左右同号姿态镜像误差 ≤ 13.8 mm（URDF 微小不对称所致）。

**转换命令**（改完 URDF 必须重转 USD）：

```bash
python scripts/tools/prepare_wheel_leg_v1_urdf.py
bash scripts/tools/convert_wheel_leg_urdf.sh
```

**随之调整的 Flat 训练配置**（`wheel_leg_task/env_cfg.py:WheelLegV1FlatEnvCfg`）：

- `height_range`：设为 `[0.16, 0.39]`（160 mm 蹲低 ~ 390 mm 机械上限，base_link 到地面）；
  原 `[0.20, 0.42]` 上界不可达，会导致身高奖励恒 0 + 平方重罚；
- `default_height_cmd`：0.32（区间内偏上）；
- `leg_joint_pair_pos_diff`：0 → **-1.0**（左右腿镜像对称惩罚，资产统一符号后此项才有意义）；
- 普通指令范围：`lin_vel_x ±2.7 → ±1.2`，`ang_vel_z ±2π → ±1.0`（先学站住/慢走，
  自旋/冲刺仍由 `special_modes` 在 iteration 2000/3000/4000 后加入）。

---

## 1. 三层架构总览

### 1.1 继承链

```
gym.make("Robotics-Wheel-Leg-V1-*")
        │  entry_point = wheel_leg_task.env:WheelLegV1Env
        ▼
WheelLegV1Env            (wheel_leg_task/env.py:98)        ← 任务层
        │ super()
WheelLegTerrainEnv       (wheel_leg_terrain/env.py:38)     ← 地形命令层
        │ super()
WheelLegBaseEnv          (wheel_leg_base/env.py:41)        ← RL 基座
        │ super()
Isaac Lab DirectRLEnv
```

配置类的继承链是对应的（`@configclass`）：

```
WheelLegV1FlatEnvCfg                (task, env_cfg.py:490)   ← 权重/地形/课程在这里
        │
WheelLegTerrainFlatEnvCfg           (terrain, env_cfg.py:397) ← 地形命令/延迟/噪声/帧堆叠
        │
WheelLegFlatEnvCfg                  (base, env_cfg.py:663)   ← 机器人/弹簧字段
        │
WheelLegBaseFlatEnvCfg              (base, env_cfg.py:375)   ← 平地基础参数
        │
WheelLegBaseEnvCfg                  (base, env_cfg.py:224)   ← sim/scene/终止/基础奖励
        │
DirectRLEnvCfg
```

### 1.2 三层职责速查

| 层 | 文件 | 负责 | 不负责 |
|---|---|---|---|
| base | `wheel_leg_base/env.py` (5700 行) `wheel_leg_base/env_cfg.py` (695 行) | 场景搭建、关节/连杆索引、动作解码与执行、观测拼接（policy/critic）、**全部 `rew_*` 奖励公式**、门控、终止、重置、域随机化事件、噪声/延迟/帧堆叠/特权观测 | 具体任务参数、地形生成器、奖励权重 |
| terrain | `wheel_leg_terrain/env.py` (380 行) `wheel_leg_terrain/env_cfg.py` (790 行) | `TerrainCommandManager` 实例化与调用、按地形覆盖命令、地形类型 marker、V13 域随机化表、观测延迟/噪声/帧堆叠的默认值配置 | 本体动力学/奖励公式 |
| task | `wheel_leg_task/env.py` (1336 行) `wheel_leg_task/env_cfg.py` (2350 行) `wheel_leg_task/cfg_utils.py` | 15 个 gym 任务注册、任务参数表与**奖励权重 `rewards`**、地形生成器选择、越界重置、高度课程、速度轨迹录制（CSV+HTML）、简单根复位、算法变体观测空间 | 通用 RL 逻辑 |

### 1.3 状态机（辅助模块，跨层）

状态机不在三层目录内，而在 `wheel_leg_v1/state_machines/`，由 base 构建、terrain 的命令钩子驱动：

- `manager.py:WheelLegStateMachineManager`：把 airborne / jump_takeoff / step_up / stair 串成一条链。
- base 在 `_build_state_machine_manager`（`env.py:1914` 附近）创建，并在每步调用 `on_observation_step`、`on_command_updated`、`apply_reward_term_scales`、`apply_done_masks` 等。
- 各任务开关：Flat-v0 关；Flat-v1/Rough-v1 开 airborne（`_apply_v14_airborne_landing_precontact_cfg` / `_apply_v14_rough_runtime_cfg`）；Rough-v0 打开后又显式关闭（`env_cfg.py:1830-1834`）。

---

## 2. base 层详解（wheel_leg_base）

### 2.1 职责

1. **场景**：机器人 USD、接触传感器、高度扫描仪、地形导入器、并行环境克隆（`_setup_scene`，`env.py:2708`）。
2. **机器人索引**：腿/轮关节与连杆索引、接触索引、材质索引（`__init__`，`env.py:1960`；关节角色在 2045-2066）。
3. **动作**：动作拆分（腿/轮）、解码（位置目标 / 速度目标）、限幅、动作噪声与延迟、执行（`_pre_physics_step` 2815 / `_apply_action` 2843）。
4. **观测**：policy / critic 拼接、缩放裁剪、帧堆叠、观测延迟/噪声（`_get_observations` 3104）。
5. **奖励引擎**：所有 `rew_*` 公式 + 门控 + 状态机后处理（`_get_rewards` 3478，`_postprocess_reward_terms` 4069）。
6. **终止/重置**：跌倒/触地/数值异常/姿态超限（`_get_dones` 4915）；重置位姿、命令、课程、随机化（`_reset_idx` 5034、`_custom_reset_random` 5368）。
7. **域随机化**：`EventCfg`（`env_cfg.py:88`）：质量、质心、惯量、摩擦、外力、增益、初始姿态。

### 2.2 关键方法索引

| 方法 | 行号 | 作用 |
|---|---|---|
| `__init__` | 1960 | 建立全部张量/索引/缓冲区；解析 `_legs_act_idx`、`_wheel_idx`、`_actuate_idx` 等 |
| `_setup_scene` | 2708 | Articulation + ContactSensor + RayCaster + 地形 + 场景克隆 |
| `_pre_physics_step` | 2815 | 存动作、低通滤波、拆腿/轮动作、解码腿位置、缩放轮动作 |
| `_apply_action` | 2843 | 腿 `set_joint_position_target`、轮 `set_joint_velocity_target`（或力矩）、延迟/噪声、禁用环境清零 |
| `_get_observations` | 3104 | 拼 policy（35 维）与 critic（78 维），含特权观测、history |
| `_get_rewards` | 3478 | 计算全部奖励项（见第 5 节） |
| `_postprocess_reward_terms` | 4069 | 交给状态机 manager 做奖励缩放 |
| `_get_dones` | 4915 | 接触/姿态/数值/观测异常/超时判定 |
| `_reset_idx` | 5034 | 重置全流程 |
| `_custom_reset_random` | 5368 | 随机腿长/腿角 IK 复位（Wheel_leg_V1 已关闭，改 task 层简单复位） |
| `_apply_spring` | 5558 | 虚拟弹簧力控（`use_spring=False`，空操作） |
| `_inverse_kinematics` | 5608 | 五连杆 IK（Wheel_leg_V1 不使用） |

### 2.3 动作 / 观测合同（wheel_leg_v1）

- **动作 6 维**：`[L_joint1, L_joint2, R_joint1, R_joint2 | L_joint3, R_joint3]`
  - 腿 4 维：位置增量，`target = default_joint_pos + leg_action_scale(0.5) * action`。
  - 轮 2 维：速度目标，`target = wheel_vel_action_scale(10.0) * action`，裁剪到 `max_wheel_vel*1.5 = 150 rad/s`。
- **policy 观测 35 维**（`WheelLegV1FlatEnvCfg`）：
  `command(3) + height_cmd(1) + root_ang_vel_b(3) + projected_gravity_b(3) + joint_pos(6) + joint_vel_leg(4) + joint_vel_wheel(2) + actions(6) + ctrl_mode_obs(7)`。
- **critic 观测 78 维**：policy 35 + `root_lin_vel_b(3) + obs_height(1) + privileged_extra(39)`（关节增益/力矩/延迟/轮速/接触/质量/材质等）。
- **频率**：`sim.dt=1/200`、`decimation=4` → 控制 50 Hz；episode 20 s。
- **观测延迟/噪声**：`use_obs_delay=True`（1~4 步），`use_act_delay=True`（1~3 步），传感器均匀噪声（角速度 ±0.25、姿态 ±0.05、关节角 ±0.025 等）。

### 2.4 对 wheelbipe 专属功能的"退化"处理

Wheel_leg_V1 无云台/弹簧/被动腿/guide/五连杆被动关节，base 里相关索引置空或关闭，逻辑靠守卫跳过：

- `_front2/3/4_joint_idx`、`_rear2_joint_idx = []`；`_legs_inact_idx = []`。
- `use_spring=False` → `_spring_idx=None`，`_apply_spring` 空操作。
- `use_leg_random_start=False` → 不用五连杆 IK 摆腿（由 task 层 `_simple_root_reset` 代替）。
- 云台相关索引仅在 `use_gimbal=True` 时查询。

---

## 3. terrain 层详解（wheel_leg_terrain）

### 3.1 职责

terrain 层**不改动力学、不加奖励公式**，只做"地形感知的命令翻译"：

1. **地形命令管理**：`TerrainCommandManager`（`wheel_leg_terrain/env.py:56` 实例化）按机器人所在 sub-terrain 覆盖：
   - `lin_vel_x / lin_vel_y / ang_vel_z`（速度指令）
   - `height_range`（身高指令范围）
   - `reset_heading_axis_aligned_only`、`disable_predefined_reset_air/ground`、`disable_jump_takeoff`、`disable_special_mode` 等开关。
2. **地形类型管理**：`TerrainTaskManager`（base 侧）提供 `get_terrain_name_mask`，供状态机按地形名允许/禁止；terrain 层负责颜色 marker 可视化。
3. **命令更新调用链**（`wheel_leg_base/env.py:_on_command_updated` 5635 附近）：

```
_on_command_updated
 ├─ _apply_jump_takeoff_permission_command
 ├─ 非朝向环境零速掩码
 ├─ _before_state_machine_command_updated()   ← terrain: 应用地形覆盖（切换时重采）
 ├─ state_machine_manager.on_command_updated() ← 状态机改写 command/height_cmd
 ├─ _apply_predefined_reset_ground_command_override()
 ├─ _after_state_machine_command_updated()    ← terrain: 再应用一次（不重采）
 └─ state_machine_manager.apply_command_overrides()
```

4. **配置贡献**：`EventCfgV13`（`env_cfg.py:163`，V13 域随机化表）、观测延迟/噪声/帧堆叠默认值、`terrain_command_overrides` 占位、`airborne_state_machine_cfg`/`wheel_forward_scan_cfg` 默认值。

### 3.2 关键方法索引

| 方法 | 行号 | 作用 |
|---|---|---|
| `WheelLegTerrainEnv.__init__` | 42 | 建 manager、marker、初始化命令状态 |
| `_setup_terrain_type_marker` / `_update_terrain_type_marker` | 73 / 116 | Play 时按地形着色球标记 |
| `_initialize_terrain_command_state` | 149 | 初始化时同步一次地形命令 |
| `_sample_height_command`（覆写） | 316 | 委托 manager 采样身高指令 |
| `_resample_custom_cmd`（覆写） | 336 | 周期重采时同步地形、重采覆盖、关特殊模式 |
| `_apply_terrain_command_overrides_for_current_step` | 347 | 核心：把当前地形覆盖写进 `self.command` / `height_cmd` |
| `_before / _after_state_machine_command_updated` | 368 / 372 | 挂在状态机命令链前后 |
| `_on_command_updated` | 376 | 上级回调 + 刷新 marker |

---

## 4. task 层详解（wheel_leg_task）

### 4.1 职责

1. **gym 注册**：`wheel_leg_task/__init__.py` 注册 15 个任务（8 训练 + 7 Play）。
2. **参数表**：`WheelLegV1FlatEnvCfg` 及其变体，定义奖励权重、地形生成器、指令范围、特殊模式、观测编码、算法观测空间。
3. **粗糙地形高度课程**：`_apply_rough_height_offset_curriculum`（`env.py:300`，由训练 iteration 外推推进，默认关闭）。
4. **速度轨迹录制**：`_record_velocity_trace` 等（`env.py:387-568`），Play 时导出 CSV + 交互式 HTML。
5. **简单根复位**：`_simple_root_reset`（`env.py:585`），把车抬到 `simple_reset_root_height=0.35 m` + 随机 yaw 后自然落下。
6. **越界重置**：`_get_rough_terrain_boundary_time_out`（`env.py:613`）按地形尺寸判定越界，按 time_out（不算摔倒）处理。
7. **配置工具**：`cfg_utils.py` 提供维度常量、地形命令覆盖表、地形生成器装配函数等。

### 4.2 任务变体一览

| 配置类 | 行号 | 说明 |
|---|---|---|
| `WheelLegV1FlatEnvCfg` | 490 | 平地基类（PPO），状态机关、绝对高度、轮速控制 |
| `WheelLegV1FlatEnvCfg_v1` | 1535 | 平地 + 腾空落地预训练（开 state machines） |
| `WheelLegV1FlatEnvCfg_v2` | 1241 | 平地 + 小陀螺平移（云台相关配置保留但本车无云台，实际退化为平移模式） |
| `WheelLegV1RoughEnvCfg` | 1751 | 粗糙地形 + 自旋/冲刺（`RM_ROTATION_TERRAINS_CFG_99`） |
| `WheelLegV1RoughEnvCfg_v1` | 1862 | 粗糙地形跑场（`RM_ROUGH_TERRAINS_CFG`，开空中重置） |
| `WheelLegV1FlatDreamWaqEnvCfg` | 1948 | DreamWaQ（policy 28 / critic 71 / policy_hist 5 帧） |
| `WheelLegV1FlatHIMEnvCfg` | 2023 | HIMLoco（同上，历史 5 帧 + 课程） |
| `WheelLegV1FlatNP3OBarlowEnvCfg` | 2093 | NP3O（on_constraint 351 维 + 5 约束通道） |
| 各 `*_Play` | — | 演示：开调试可视化、固定地形、轨迹录制 |

### 4.3 cfg_utils.py 关键内容

| 内容 | 说明 |
|---|---|
| 维度常量 | `V14_BASE_POLICY_OBS_DIM=28`、`V14_BASE_PRIVILEGED_OBS_DIM=32`、`V14_WHEEL_LEG_PRIVILEGED_OBS_DIM=71`、`V14_WHEEL_LEG_PRIV_LATENT_DIM=43`、`V14_NP3O_ON_CONSTRAINT_V1_DIM=351` |
| 观测裁剪/缩放表 | `V14_BASIC_OBS_CLIP/SCALE`、`V14_EXTRA_OBS_CLIP/SCALE` |
| 关节/连杆顺序 | `V14_ORDERED_LEG_JOINT_NAMES = (L_joint1, L_joint2, R_joint1, R_joint2)` |
| 地形命令覆盖表 | `V14_ROUGH_TERRAIN_COMMAND_OVERRIDES`、`V14_ROTATION_TERRAIN_COMMAND_OVERRIDES_1/2` |
| 粗糙地形装配 | `_apply_v14_rough_runtime_cfg`（565 附近）：换生成器、开状态机、装配命令覆盖 |
| 高度课程默认参数 | `V14_ROUGH_HEIGHT_OFFSET_CURRICULUM_DEFAULT_CFG`（`enabled=False`，interval=500，num_levels=11） |

---

## 5. 奖励体系

### 5.1 计算位置 vs 权重位置

- **公式**：全部在 `wheel_leg_base/env.py:_get_rewards`（3478）。函数内所有以 `rew_` 开头的局部变量会被自动收集成字典（变量名去掉 `rew_` 前缀即奖励项名）。
- **权重**：在 task 层的 `rewards = OrderedDict(...)`（`wheel_leg_task/env_cfg.py:913`）。没写进字典的项一律按 0 处理。
- **装配顺序**（`_get_rewards` 末尾）：
  1. 收集 `rew_*` → `_postprocess_reward_terms`（状态机缩放/新增，task 层还会加小陀螺专属项）；
  2. 每项 × `step_dt`（0.02 s，所以权重是"每秒"量纲）；
  3. 每项 × `cfg.rewards[key]`；
  4. NaN/Inf 清洗、数值异常环境清零。
- **门控**：`vel_upright_gate * vel_orientation_x/y_gate * vel_height_gate` 会乘到速度追踪类奖励上；`height_upright_gate` 乘到身高追踪类上（各任务按需开，Flat 默认关，NP3O 开身高门控）。

### 5.2 奖励项全表（Flat 默认权重；Rough 差异在备注）

约定：`pgb`=投影重力，`v`=机体系速度，`cmd`=指令，`q`=关节角，`τ`=关节力矩，`err`=指令−实测。

#### A. 存活/终止
| 项 | 公式（简） | Flat 权重 | 备注 |
|---|---|---|---|
| `termination` | `reset_terminated` | -200 | 摔倒大罚；腾空状态机可再缩放 |

#### B. 正则化（平滑/省电/护电机）
| 项 | 公式（简） | Flat 权重 | 备注 |
|---|---|---|---|
| `leg_joint_acc` | Σ `joint_acc²`（腿 4 关节） | -5e-7 | 腿别猛甩 |
| `leg_joint_vel` | Σ `joint_vel²`（腿） | -5e-3 | |
| `leg_joint_pair_pos_diff` | Σ `wrap( q_L − q_R )²` | -0.0 | 左右对称（当前关） |
| `joint_torque` | Σ `τ²`（6 执行关节） | -1e-4 | Rough 调为 -1e-5 |
| `wheel_acc` | Σ `joint_acc²`（轮） | -1e-8 | |
| `wheel_vel` | Σ `joint_vel²`（轮） | -1e-5 | |
| `wheel_power` | Σ `clamp(τ_w·v_w, min=0)` | -1e-4 | 正功率（耗电）惩罚；Rough 调为 -1e-5 |
| `wheel_air_spin` | Σ `v_wheel² · 离地` | 0 | 腾空空转惩罚（当前关） |
| `lin_vel_z` | `v_z²` | -0.5 | 别上下颠 |
| `ang_vel_xy` | Σ `ω_xy²` | -0.05 | 车身别乱转 |
| `action_rate` | Σ `(a−a_prev)²` | -0.01 | |
| `action_smoothness_leg` | Σ 二阶差分²（腿） | -0.05 | |
| `action_smoothness_wheel` | Σ 二阶差分²（轮） | -0.01 | |

#### C. 姿态
| 项 | 公式（简） | Flat 权重 | 备注 |
|---|---|---|---|
| `flat_orientation_y` | `(σ·pgb_x)²` | -0.0 | 直接用重力投影（当前关） |
| `flat_orientation_y_v` | `((A·exp(−vx²/σ)+b)·pgb_x)²` | -2.0 | 随速度变化的俯仰惩罚 |
| `flat_orientation_y_exp` | `exp(−pgb_x²/σ)` | +1.0 | 俯仰奖励（越平越好） |
| `flat_orientation_x` / `_v` / `_exp` | 同上的横滚版本 | -0.0 / -2.0 / +1.0 | |
| `flat_pitch_l1/tanh`、`flat_roll_l1/tanh` | 见代码 | 未启用 | 备选姿态项 |

#### D. 任务追踪（主线）
| 项 | 公式（简） | Flat 权重 | 备注 |
|---|---|---|---|
| `track_lin_vel_xy` | `exp(−err²/σ)`，带姿态/身高门控 | +1.0 | Rough-v1 提到 +1.25 |
| `track_lin_vel_xy_tight` | 同上，σ 更小 | 0.0 | |
| `track_lin_vel_xy_square` | `(σ·err)²` | -1.0 | 误差平方罚 |
| `track_ang_vel_z` | `exp(−err²/σ)`，带门控 | +1.0 | |
| `track_ang_vel_z_square` | `(σ·err)²` | -1.0 | |
| `stand_still` | `Σv_xy²·mask + ωz²·mask` | -0.0 | |
| `stand_still_lin_vel` | `Σ|v_xy|·mask` | -1.0 | 指令≈0 时要求站住 |
| `stand_drift` | `(σ·max(‖p_xy−p_ref‖−deadband,0))²·mask` | -2.0 | ★指令≈0 时净水平位移罚：防慢慢漂走，允许平衡微动（deadband=0.1 m, σ=1.0；`_stand_ref_pos_w` 由 `_simple_root_reset` 记录） |
| `track_height_exp` | `exp(−h_err²/σ)` | 0.0 | |
| `track_height_exp_soft` | `exp(−h_err²/σ_soft)` | 0.0 | |
| `track_height_exp_tight` | `exp(−h_err²/σ_tight)` | +1.0 | ★身高跟踪主项 |
| `track_height_square` | `(σ·h_err)²` | -1.0 | |
| `track_height_exp_both_wheels_contact` | exp × 双轮着地 | 0.0 | |

#### E. 腿型/防劈叉/接触（轮腿特有）
| 项 | 公式（简） | Flat 权重 | 备注 |
|---|---|---|---|
| `no_fork` | 布尔：两轮 x 向距离 > `no_fork_distance` | -1.0 | 防劈叉 |
| `no_fork_square` | `(Δx·σ)²` | -1.0 | |
| `no_fork_exp` / `no_fork_z_exp` | `1−exp(−over/σ)` | -0.0 | 当前关 |
| `wheel_motor_z_axis_align_exp(_tight)` | `exp(−误差²/σ)`，误差=腿方向与重力夹角 | base 默认 +0.1 | task Flat 重写未含，实际 0 |
| `undesired_contact` | 不该碰的部件（base_link / L_link1/2 / R_link1/2）接触力 > 阈值 | -2.0 | |
| `desired_contact` | 轮子接触力峰值统计 | 未启用 | |

> 完整项（含 `_huge_gap`、`_pen_high_speed`、`foot_bound_*`、`l/r_leg_ang_exp`、`rear2_rear1_joint_pos_limits*` 等）见 `_get_rewards` 3478-4065，未列入 `rewards` 的即为 0。

### 5.3 三个任务组的权重差异

| 项 | Flat | Rough-v0 | Rough-v1 |
|---|---|---|---|
| `wheel_power` | -1e-4 | -1e-5 | -1e-5 |
| `joint_torque` | -1e-4 | -1e-5 | -1e-4→-1e-5 |
| `track_lin_vel_xy` | 1.0 | 1.0 | **1.25** |
| `stand_still_lin_vel` | -1.0 | -1.0 | -1.0 |
| 特殊模式（自旋/冲刺） | 按 iteration 3000/4000 开 | **iteration 0 开** | 同 v1 + 空中指令重采 |
| 姿态/身高门控 | 默认关 | 关 | 关 |

### 5.4 状态机对奖励的修改

`_get_rewards` → `_postprocess_reward_terms` → `WheelLegStateMachineManager.apply_reward_term_scales`。各状态机可：
- 按状态覆盖某些项权重（如 airborne 期间 `undesired_contact ×25`、`termination ×3`）；
- 新增状态专属奖励项（如 stair 成功/失败奖励、jump 阶段奖励）。

Wheel_leg_V1 中该机制仅在开启状态机的任务（Flat-v1、Rough-v1）生效；Flat-v0 无状态机。

---

## 6. 地形设计

### 6.1 Flat（平地）

- `terrain_type="plane"`（`WheelLegBaseEnvCfg.terrain`），无生成器。
- 物理材质 static/dynamic=1.0，restitution=0，combine=multiply。
- 机器人出生即平面，高度用绝对高度口径（`use_absolute_height=True`）。

### 6.2 Rough-v0（小陀螺/平移）生成器

`RM_ROTATION_TERRAINS_CFG_99`（`terrains.py:672`）：

- `curriculum=True`，`size=(9,9) m`，`num_rows=10`（10 级难度行）× `num_cols=10`，`border_width=5 m`。
- `difficulty_range=(0,1)`；`horizontal_scale=0.1`，`vertical_scale=0.005`。
- 子地形（当前启用）：

| key | 类型 | 关键参数 |
|---|---|---|
| `tiny_step_rot` | 自定义网格条 (`MeshCustomGridBarsTerrainCfg`) | 条宽 0.05~0.2 m，条高 0.01~0.03 m，条数随难度 2~4 |
| `slope_for_rm_low` | 金字塔斜坡 | 斜率 0.05~0.15，无边平台，border 1 m |
| `inv_slope_for_rm_low` | 反金字塔斜坡 | 同上，inverted |
| `stair_slope_for_rm_low` | 金字塔阶梯 | 步高 0.005~0.015 m，步宽 0.1 m，平台 1 m |
| `inv_stair_slope_for_rm_low` | 反金字塔阶梯 | 步高 0.005~0.015 m，平台 2 m |
| `plane_for_rm_rot` | 平面瓦片 | proportion 0.3 |
| `random_uniform_for_rm` | 随机粗糙 | 噪声 0~0.03 m，步长 0.005 m |

- 出生难度：生成器开 `curriculum=True` 且 `max_init_terrain_level` 未设置 → Isaac Lab 会在 **0~num_rows-1 全难度行随机出生**（不是从简到难）；仓库自带的高度课程默认 `enabled=False`（要渐进难度需打开 `rough_height_offset_curriculum_cfg`）。

### 6.3 Rough-v1（跑场）生成器

`RM_ROUGH_TERRAINS_CFG`（`terrains.py:602`）：

- `curriculum=False` → 所有地块按 `difficulty_range=(0,1)` 随机采样难度；`num_rows=10 × num_cols=13`，size 9×9 m，border 10 m。
- 子地形：

| key | 类型 | 关键参数 |
|---|---|---|
| `low_speed_stair_for_rm` | 大台阶 | 步高 0.15~0.35 m，平台 3 m |
| `tiny_step` | 小台阶 | 见定义 |
| `slope_for_rm_high` | 金字塔斜坡 | 斜率 0.15~0.35 |
| `inv_slope_for_rm_low` | 反斜坡 | 0.05~0.15 |
| `high_speed_stair_for_rm` | 高速大台阶 | 步高 0.15~0.40 m |
| `stair_slope_for_rm_high` | 金字塔阶梯 | 步高 0.015~0.032 m |
| `inv_stair_slope_for_rm_low/high` | 反阶梯 | 0.005~0.015 / 0.015~0.032 m |
| `plane_for_rm` | 平面 | |
| `random_uniform_for_rm` | 随机粗糙 | 0~0.03 m |
| `cliff_inv_stair_slope_short_for_rm` | 断崖反阶梯 | 台阶 0.025~0.032 m，落差 0.3~0.4 m |

> Rough-v1 的 `__post_init__` 只调用 `_apply_v14_rough_runtime_cfg`，若 `rough_terrain_generator_cfg` 未设置则回退到 `RM_ROUGH_TERRAINS_CFG`。

### 6.4 Play 地形

- Flat Play：仍是平面。
- Rough Play 系列：`RM_ROUGH_TERRAINS_PLAY_CFG`（`terrains.py:734`，1×1 地块），使用
  `cliff_inv_stair_slope_short_for_rm_play`（固定台阶 0.03 m，落差 0.3~0.4 m，平台 4 m），
  并配合固定前冲指令（Rough-v0 Play 为 2.2 m/s 飞坡）做演示。

### 6.5 地形命令覆盖

`cfg_utils.py` 中定义按地形名生效的 `TerrainCommandOverrideCfg` 字典（`V14_ROUGH_TERRAIN_COMMAND_OVERRIDES`、`V14_ROTATION_TERRAIN_COMMAND_OVERRIDES_1/2`），可覆盖速度范围、身高范围、朝向锁定与状态机开关。运行时由 `TerrainCommandManager` 应用；未列出的地形沿用 `commands` 的全局采样范围。

### 6.6 出生/越界/课程

- 出生：`reset_base` 事件随机 roll/pitch ±0.15 rad、yaw ±π；`_simple_root_reset` 抬到 0.35 m 后落下（Rough-v1 还开 `predefined_reset_air` 空中出生）。
- 越界：Rough 任务开 `rough_terrain_boundary_reset_cfg`，超出地形半边界减 margin 0.5 m 即按 time_out 重置。
- 高度课程：`V14_ROUGH_HEIGHT_OFFSET_CURRICULUM_DEFAULT_CFG`（`enabled=False`）——打开后按训练 iteration 每 500 轮升一级，共 11 级，并在 `_apply_rough_height_offset_curriculum` 中把 env 搬到对应难度行。
- 平地站立起步课程：`CurriculumCfgV14Stand`（`env_cfg.py:485`）——① 托举力 60→30→0 N；② 高度范围 `[0.30,0.32]` 逐级展宽到 `[0.25,0.39]`；③ 腿振荡惩罚逐级收紧；④ `command_velocity_progression` 速度指令分档：阶段0 全体零指令（`rel_standing_envs=1.0`，并在构造时禁用旧 `zero_cmd`）→ 阶段1 小速度 ±0.4 m/s / ±0.5 rad/s → 阶段2 常规 ±1.2 m/s / ±1.0 rad/s；晋级条件 = 最少轮数（400/300）+ 窗口平均存活（≥10/12 s）+ `track_height_exp_tight` 窗口均值 ≥0.4。

---

## 7. 观测/动作/奖励的"归属"速查

| 内容 | 定义在 | 调用/生效 |
|---|---|---|
| 动作解码与执行 | base `_pre_physics_step` / `_apply_action` | task `_apply_action` 追加云台（本车无） |
| policy/critic 拼接 | base `_get_observations` | task 可改 ctrl_mode_obs（spin 模式）；算法变体在 task cfg 改维度 |
| 奖励公式 | base `_get_rewards` | 状态机后处理；task 加模式下专属项 |
| 奖励权重 | task `rewards` | base 装配时读取 |
| 地形命令覆盖 | terrain `TerrainCommandManager` + task `terrain_command_overrides` | terrain 钩子链在每次命令更新时应用 |
| 地形生成器 | task `rough_terrain_generator_cfg` → `cfg_utils._apply_v14_rough_runtime_cfg` | base `_setup_scene` 创建 |
| 域随机化 | base `EventCfg` / terrain `EventCfgV13` / task `EventCfgV14` | 任务最终用 `EventCfgV14`（继承 `EventCfg`） |

---

## 8. 训练与验证

```bash
conda activate isaaclab
cd /home/noir/Documents/workspace/example/wheeled-legged_RL

# 平地 PPO
python scripts/rsl_rl/train.py --task=Robotics-Wheel-Leg-V1-Flat-v0 \
    --num_envs=4096 --max_iterations=20000 --headless --device=cuda:0

# 粗糙地形
python scripts/rsl_rl/train.py --task=Robotics-Wheel-Leg-V1-Rough-v0 \
    --num_envs=4096 --max_iterations=20000 --headless --device=cuda:0

# 算法变体
python scripts/rsl_rl/train.py --task=Robotics-Wheel-Leg-V1-Flat-DreamWaQ-v0 ...
python scripts/rsl_rl/train.py --task=Robotics-Wheel-Leg-V1-Flat-HIM-v0 ...
python scripts/rsl_rl/train.py --task=Robotics-Wheel-Leg-V1-Flat-NP3OBarlow-v0 ...

# 可视化 / 键盘控制
python scripts/view_robot.py --task=Robotics-Wheel-Leg-V1-Flat-Play-v0 --num_envs=1
./run_gui.sh python scripts/rsl_rl/play.py --task=Robotics-Wheel-Leg-V1-Flat-Play-v0 \
    --num_envs=1 --checkpoint=<model.pt> --keyboard
```

---

## 9. 已知缺口 / 后续可做

1. **摩擦实机标定**：当前随机化范围是经验值；建议实机测 μ–滑移曲线后收窄，并区分地面材质。
2. **轮子碰撞体**：mesh→convex hull，可考虑换解析圆柱/胶囊并校准轮半径与接触刚度。
3. **地形课程默认关**：要渐进难度需打开 `rough_height_offset_curriculum_cfg`，并考虑设置 `max_init_terrain_level` 控制出生难度。
4. **Rough-v0 状态机被显式关闭**：若要练跳跃/爬台阶，参考 Rough-v1 或 Flat-v1 的开关组合。
5. **Play 随机化未关**：演示复现性要求高时，需手动禁用对应 `EventTerm`。
6. **v2 的小陀螺专属奖励项**：本车无云台，相关项处于不激活状态；若后续加云台需复核 obs/act 合同。

---

## 10. 参考行号索引（2026-09-11 快照）

| 内容 | 位置 |
|---|---|
| base 环境类 | `wheel_leg_base/env.py:41`、`__init__:1960`、`_apply_action:2843`、`_get_observations:3104`、`_get_rewards:3478`、`_get_dones:4915`、`_reset_idx:5034` |
| base 配置 | `wheel_leg_base/env_cfg.py:224/375/663`，`EventCfg:88`，`rewards:591` |
| terrain 环境类 | `wheel_leg_terrain/env.py:38`、TCM:56、override 应用:347 |
| terrain 配置 | `wheel_leg_terrain/env_cfg.py:163/397` |
| task 环境类 | `wheel_leg_task/env.py:98`、课程:300、轨迹:501、简单复位:585、越界:613 |
| task 配置 | `wheel_leg_task/env_cfg.py:490`、`EventCfgV14:106`、`rewards:913`、Rough-v0:1751、Rough-v1:1862、算法变体:1948/2023/2093 |
| 工具/常量 | `wheel_leg_task/cfg_utils.py`（维度常量、overrides:461/499、rough 装配:565 附近） |
| 地形生成器 | `manager/mdp/isaaclab/terrains.py:602/672/734` |
| 状态机 | `wheel_leg_v1/state_machines/manager.py` 及各状态文件 |

> 注意：本仓库文件可能被并发修改；改动代码后请以函数名/类名重新定位，必要时更新本文档第 10 节。
