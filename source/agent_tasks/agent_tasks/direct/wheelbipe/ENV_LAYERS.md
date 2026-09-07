# Wheelbipe 环境三层继承结构与函数全览

> 回答两个问题:① 三层环境类(V14 → V13 → wheelbipe25_v3)各自负责什么、新增了什么;
> ② 每一层里面的所有函数分别是什么作用。
>
> 三层是**继承链**,不是三个独立程序。跑 V14 时只有一个环境对象、一次训练循环,
> 每步执行的是"洋葱式"串联起来的一份逻辑:V14 干自己新增的,`super()` 往下委托给
> V13 / wheelbipe25_v3 / Isaac Lab 的 `DirectRLEnv` 干基座部分。

```
DirectRLEnv(Isaac Lab 框架:step 主循环、物理仿真)
   └── Wheelbipe25V3Env(基座层:腿/轮电机、奖励公式库、观测拼接、命令生成、重置/随机化)
         └── WheelbipeV13Env(地形命令层:TerrainCommandManager 按地形覆盖命令)
               └── WheelbipeV14Env(云台/课程/录制层:云台控制、地形课程、小陀螺平移、轨迹录制)
```

一次训练 step 中框架的调用顺序(每层钩子都经 `super()` 串成一条链):

```
_pre_physics_step → _apply_action → 物理仿真推进 → _get_dones → _get_rewards
→ _reset_idx(有 done 时) → _get_observations
```

各文件位置(均在 `agent_tasks/direct/wheelbipe/` 下):

| 层 | 文件 | 行数 | 函数数 |
|---|---|---|---|
| 基座 | `wheelbipe25_v3/env.py` | ~5700 | 197 |
| 地形命令 | `wheelbipe_V13/env.py` | 379 | 20 |
| 云台/课程 | `wheelbipe_V14/env.py` | 1322 | 57 方法 + 4 模块函数 |

---

# 一、wheelbipe25_v3:基座层(所有轮腿任务的公共实现)

## 1.1 这一层负责什么

`class Wheelbipe25V3Env(DirectRLEnv)`,直接继承 Isaac Lab 框架,是 V13/V14 的地基。
它实现了轮腿机器人的全部底层环境机制,**不包含任何 V13/V14 的任务特化逻辑**:

- **执行器控制**:腿关节位置伺服、轮关节速度/力矩伺服、弹簧(棱柱关节)助力、动作低通滤波与噪声;
- **奖励公式库**:几十个 `rew_*` 项(速度/高度/航向跟踪多档、姿态、能耗、平滑、对称、接触、限位),
  带姿态门控与高度门控,按 `cfg.rewards` 权重加权;
- **观测拼接**:policy 与 critic 双流、Sim2Sim 历史帧堆叠与帧遮蔽、电机/IMU 延迟建模、
  高度射线扫描、特权观测(质量/惯量/材料/延迟 lag 等);
- **命令生成**:XY 速度 + 偏航 + 高度命令;特殊高度模式(阶跃/正弦波)、起跳权限系统、
  预定义空中/地面重置、轴对齐重置朝向;
- **结束判定与重置**:接触/姿态/数值安全(NaN、离群)/观测异常终止,连续 N 步生效机制;
  重置时的姿态/速度随机化(`_custom_reset_random`)、五连杆腿逆解摆位;
- **域随机化**:质量/惯量/材料/增益随机化参数的收集与应用;
- **play 可视化**:偏航角速度箭头、轮前向地形扫描点、内置地形类型标记;
- **状态机**:`WheelbipeStateMachineManager` 状态机栈,通过一组钩子(`_before/_after_state_machine_command_updated`、
  `_postprocess_reward_terms` 等)让子类插入任务逻辑而不必重写底层。

## 1.2 核心 RL 钩子(框架每步按顺序调用)

| 钩子 | 职责 |
|---|---|
| `_pre_physics_step(actions)` | 每个控制步最先执行:清步内缓存;动作低通滤波;切分为腿/轮两段;腿动作按编码解码成关节位置命令,轮动作乘 scale |
| `_apply_action()` | 物理步进前下发:动作限幅加噪;腿关节发位置目标;轮关节发速度(或力矩)目标;初始化期零力矩;`_apply_spring()` 施加弹簧力 |
| `_get_dones()` | 结束判定:接触/姿态终止 + 超时;NaN/离群/观测异常立即终止;状态机可覆盖终止掩码;非立即终止可要求连续 N 步成立 |
| `_get_rewards()` | 奖励主体:计算全部 `rew_*` 项 → `_postprocess_reward_terms` 状态机缩放 → 按 cfg 权重加权、NaN 保护 → 刷新命令并触发命令钩子 → episode 求和 |
| `_get_observations()` | 观测拼接:更新地面估计与延迟模型;触发 `_on_command_updated`;拼装关节/命令/高度/扫描等块,clip+scale;critic 另拼特权观测;历史帧堆叠 |
| `_reset_idx(env_ids)` | 重置入口:记录终止特权观测、跑课程;清各缓冲/历史/延迟;重采样命令、朝向、高度;写默认关节状态后调 `_custom_reset_random`;重置状态机与标记 |
| `_postprocess_reward_terms(terms)` | 薄钩子:把奖励项交给状态机做任务级缩放;是子类插入奖励整形的主要通道 |
| `_on_command_updated()` | 命令后处理钩子(每步观测与奖励内的命令刷新后调用):起跳权限 → 轴对齐清零 → 状态机前后钩子与命令覆盖 |
| `_custom_reset_random(env_ids)` | 重置随机化:弹簧力、腿部初始姿态(含预定义空中/地面模式)、关节速度随机 |

## 1.3 全部函数清单

### 奖励门控参数读取(奖励用)

| 函数 | 作用 |
|---|---|
| `_get_vel_height_gate_full_error` | 读 cfg:高度门控全奖励误差阈值 |
| `_get_vel_height_gate_zero_error` | 读 cfg:高度门控零奖励误差阈值 |
| `_get_vel_height_gate_enabled` | 读 cfg:高度-速度门控是否启用 |
| `_get_vel_orientation_x_gate_enabled` | 读 cfg:x 姿态门控开关 |
| `_get_vel_orientation_x_gate_full_deg` | 读 cfg:x 姿态门控全奖励角度 |
| `_get_vel_orientation_x_gate_zero_deg` | 读 cfg:x 姿态门控零奖励角度 |
| `_get_vel_orientation_y_gate_enabled` | 读 cfg:y 姿态门控开关 |
| `_get_vel_orientation_y_gate_full_deg` | 读 cfg:y 姿态门控全奖励角度 |
| `_get_vel_orientation_y_gate_zero_deg` | 读 cfg:y 姿态门控零奖励角度 |
| `_as_reward_gate_tensor` | 把标量/张量门控值转成与参考同形的张量 |
| `_as_reward_gate_bool_tensor` | 把布尔门控值转成同形布尔张量 |

### play 模式可视化:角速度箭头

| 函数 | 作用 |
|---|---|
| `_is_play_ang_vel_z_debug_vis_enabled` | 判断 play 模式是否显示角速度箭头 |
| `_setup_play_ang_vel_z_marker` | 创建指令/实测偏航角速度箭头标记 |
| `_set_play_ang_vel_z_marker_visibility` | 设置两个角速度箭头可见性 |
| `_resolve_yaw_rate_to_arrow` | 把角速度映射为机体系 y 向箭头 |
| `_get_play_ang_vel_z_marker_positions` | 返回机器人上方的标记位置 |
| `_update_play_ang_vel_z_marker` | 每步刷新指令/实测角速度箭头 |

### play 模式可视化:状态机与轮前向地形扫描

| 函数 | 作用 |
|---|---|
| `_setup_state_machine_marker` | 经状态机管理器创建状态机标记 |
| `_update_state_machine_marker` | 经状态机管理器刷新状态机标记 |
| `_setup_wheel_forward_scan_marker` | 创建轮前向地形探测点标记 |
| `_is_wheel_forward_scan_enabled` | 读 cfg:轮前向地形探测是否启用 |
| `_get_wheel_forward_scan_cfg` | 返回轮前向扫描配置字典 |
| `_get_dynamic_wheel_forward_scan_points` | 计算两轮前向探测的查询/命中点 |
| `_get_wheel_forward_scan_points` | 返回常规轮前向扫描点(带缓存) |
| `_get_wheel_forward_stair_scan_points` | 返回短距阶梯探测扫描点 |
| `_update_wheel_forward_scan_marker` | play 模式刷新地形命中点标记 |

### 内置地形调试标记与地形任务管理

| 函数 | 作用 |
|---|---|
| `_is_builtin_terrain_debug_marker_enabled` | 判断 play 模式是否显示地形类型标记 |
| `_build_builtin_terrain_task_manager` | 创建粗糙地形名→标记映射管理器 |
| `_get_terrain_task_manager` | 惰性获取缓存的地形任务管理器 |
| `get_terrain_name_mask` | 按地形名返回各 env 的布尔掩码(公开方法) |
| `_setup_builtin_terrain_debug_marker` | 创建粗糙地形类型可视化标记 |
| `_update_builtin_terrain_debug_marker` | 每步更新机器人上方地形类型标记 |

### 高度信号与腿部姿态误差

| 函数 | 作用 |
|---|---|
| `_use_absolute_height` | 读 cfg:高度是否用绝对世界高度 |
| `_use_leg_length_height` | 读 cfg:高度是否用腿长等效高度 |
| `_use_raycast_height` | 读 cfg:高度是否用射线离地高度 |
| `_get_height_measure_wheel_radius` | 返回腿长-身高换算用轮半径 |
| `_get_leg_length_height` | 返回机身到轮关节中心的平均腿长 |
| `_apply_height_obs_clip` | 对高度观测做限幅与安全数值处理 |
| `_get_observed_height` | 返回观测/奖励统一使用的高度信号 |
| `_get_wheel_motor_z_axis_align_error_sq` | 返回腿电机轴与重力方向失配平方(奖励) |
| `_maybe_print_wheel_motor_z_axis_align_debug` | 低频打印电机轴对齐调试信息 |
| `_get_body_material_for_link` | 返回某 link 的摩擦/恢复系数材料 |
| `_get_wheel_contact_force_for_link` | 返回某轮世界系接触力及历史峰值 |
| `_maybe_print_wheel_material_debug` | 打印左右轮材料与接触力 |

### 轮相对地面高度与高度奖励参考

| 函数 | 作用 |
|---|---|
| `_get_scanner_ground_height` | 用射线扫描器估计每 env 地面高度 |
| `_get_wheel_body_height` | 返回指定轮体的世界系中心高度 |
| `_get_wheel_relative_ground_heights_raw` | 返回右/左轮相对本地地面高度 |
| `_get_current_wheel_ground_z_raw` | 由轮下扫描器返回地面 z 高度 |
| `_update_height_reward_airborne_state` | 依最新观测更新腾空状态机 |
| `_get_wheel_forward_spatial_height_diffs_raw` | 前向探测点与当前轮下高度差 |
| `_get_wheel_forward_temporal_height_diffs_raw` | 前向扫描高度相对上控制步的时间差分 |
| `_get_wheel_forward_stair_temporal_height_diffs_raw` | 短距阶梯扫描的时间差分 |
| `_get_height_reward_reference_height` | 返回 track_height 奖励用参考高度 |
| `_init_special_height_wave_state` | 初始化动态高度特殊模式的每 env 状态 |

### 高度命令特殊模式(配置与采样)

| 函数 | 作用 |
|---|---|
| `_cfg_value` | 兼容 Mapping/对象两种 cfg 取值方式 |
| `_height_command_special_modes_cfg` | 读 cfg:高度特殊模式配置块 |
| `_get_height_command_profile_cfg` | 解析模式为 step 或 wave 剖面 |
| `_height_command_special_mode_entries` | 列出启用的(索引, 名, 配置)条目 |
| `_is_height_command_special_mode_active` | 按 env 区间/迭代区间判断模式是否生效 |
| `_sample_height_command_special_phase` | 采样特殊模式的相位(可随机) |
| `_sample_height_wave_param` | 采样正弦波模式的幅值/频率等参数 |
| `_resample_height_command_special_modes` | 为选中 env 独立重采样高度特殊模式 |
| `_apply_height_command_special_modes` | 不改速度命令地套用高度特殊模式 |
| `_latch_special_height_wave` | 为波模式 env 锁存相位与起始时间 |
| `_apply_special_height_wave` | 为波模式 env 生成动态正弦高度命令 |
| `_get_effective_height_cmd` | 返回高度奖励实际使用的运行时命令 |
| `_get_observation_height_cmd` | 返回暴露给观测的高度命令 |

### 通用小工具与轴对齐重置朝向

| 函数 | 作用 |
|---|---|
| `_as_env_ids_tensor` | 把 env_ids 规范为设备上一维张量 |
| `_get_axis_aligned_reset_heading_mask` | 返回本回合锁存的轴对齐重置掩码 |
| `_get_predefined_reset_air_disabled_mask` | 返回预定义空中重置模式禁用掩码 |
| `_get_predefined_reset_ground_disabled_mask` | 返回预定义地面重置模式禁用掩码 |
| `_update_axis_aligned_reset_heading_mask` | 按回合锁存轴对齐重置朝向规则 |
| `_sample_height_command` | 为指定 env 采样高度命令 |

### 跳跃起跳权限系统

| 函数 | 作用 |
|---|---|
| `_get_jump_takeoff_permission_cfg` | 读 cfg:起跳权限配置字典 |
| `_is_jump_takeoff_permission_enabled` | 读 cfg:起跳权限机制是否启用 |
| `_normalize_permission_range_spec` | 规范化区间定义为元组列表 |
| `_sample_permission_range_spec` | 从区间规范采样数值 |
| `_is_jump_takeoff_permission_iteration_active` | 判断当前训练迭代是否允许起跳权限 |
| `_resample_jump_takeoff_permission` | 重采样哪些 env 拥有起跳权限及速度范围 |
| `_apply_jump_takeoff_permission_height_range` | 有起跳权限时覆盖高度命令范围 |
| `_apply_jump_takeoff_permission_command` | 把权限速度范围写入命令生成器 |
| `_clear_predefined_reset_ground_command_override` | 清除地面预定义重置的命令覆盖 |
| `_set_predefined_reset_ground_command_override` | 设置地面预定义重置期间的命令覆盖 |
| `_apply_predefined_reset_ground_command_override` | 每步套用地面预定义重置命令覆盖 |
| `_get_special_mode_disable_jump_takeoff_mask` | 返回特殊模式禁用起跳的掩码 |
| `_get_jump_takeoff_disabled_mask` | 汇总各类来源的起跳禁用掩码 |
| `_apply_special_mode_height_ranges` | 为特殊模式 env 覆盖采样后的高度命令 |
| `_request_special_mode_jump_takeoff` | 为启用起跳的特殊模式 env 发起起跳请求 |
| `_force_resample_commands` | 立即对选中 env 强制重采样命令 |

### 重置朝向、任务标志与额外观测块

| 函数 | 作用 |
|---|---|
| `_get_non_heading_axis_aligned_zero_mask` | 返回需把偏航命令清零的 env 掩码 |
| `_apply_axis_aligned_reset_heading` | 重置朝向吸附到配置的轴对齐角度 |
| `_record_reset_heading_target` | 缓存重置偏航为本回合目标航向 |
| `_sync_heading_command_target_to_reset_heading` | 把航向命令目标同步为重置朝向 |
| `_get_height_reward_target_height` | 返回 track_height 奖励的目标高度 |
| `_get_task_flag_obs_raw` | 返回命令观测旁的任务标志占位块 |
| `_get_policy_task_flag_obs_raw` | 返回策略可见的任务标志块 |
| `_get_policy_extra_obs_blocks` | 返回追加在动作后的命名策略观测块 |
| `_get_critic_extra_obs_blocks` | 返回特权附加前的命名 critic 观测块 |
| `_get_ctrl_mode_obs_raw` | 返回控制模式观测:状态 one-hot + 跳高目标 + 相位时间 |
| `_get_jump_takeoff_extra_obs_raw` | 旧配置兼容别名,转调 ctrl_mode 观测 |

### 步内缓存与轮运动学

| 函数 | 作用 |
|---|---|
| `_invalidate_step_caches` | 清空同一控制步内的物理量缓存 |
| `_get_root_quat_inv_and_wheel_pos_b` | 每步一次计算机体系/航向系轮位置运动学 |
| `_build_state_machine_manager` | 创建共享的 wheelbipe 运行时状态机栈 |
| `request_jump_takeoff` | 为选中 env 置起跳请求标志(外部/课程调用) |
| `_get_left_right_leg_joint_pair_indices` | 返回左右腿对称关节索引对(奖励用) |

### 构造与初始化

| 函数 | 作用 |
|---|---|
| `__init__` | 初始化全部缓冲/索引/命令/延迟/帧堆叠/状态机 |
| `_find_contact_sensor_indices` | 按正则在接触传感器 body 中找索引 |
| `_build_material_mapping` | 建立 body 名→材料属性(23 维)索引映射 |
| `_get_material_indices` | 按名称模式返回材料的 body 索引 |
| `_init_static_index_layouts` | 缓存仅依赖 cfg/拓扑的索引与计数布局 |
| `_init_privileged_extra_obs_layout` | 计算特权附加观测各分块维度布局 |
| `_set_play_height_scanner_debug_vis` | play 模式开启所有高度射线调试可视化 |
| `_setup_scene` | 搭建场景:机器人/地形/克隆/灯光 |

### 动作处理与 RL 执行钩子

| 函数 | 作用 |
|---|---|
| `_low_pass_action_filter` | 对动作做一阶低通滤波 |
| `_get_leg_policy_action_dim` | 返回腿部策略动作维度 |
| `_split_policy_actions` | 把策略动作切分为腿/轮两段 |
| `_decode_leg_policy_actions` | 按编码解析腿动作为关节位置命令 |
| `_get_policy_action_slices` | 返回腿/轮动作切片 |
| `_pre_physics_step` | RL 钩子:清缓存、滤波、切分解码动作(见 1.2) |
| `_apply_action` | RL 钩子:限幅加噪后下发腿/轮/弹簧控制(见 1.2) |

### 观测组装

| 函数 | 作用 |
|---|---|
| `_update_obs` | 更新延迟模型:电机/IMU 信号按 lag 延迟 |
| `_debug_print_undesired_contacts` | play 模式低频打印非期望接触 |
| `_get_wheel_contact_force_peaks` | 返回接触历史窗内每轮最大接触力 |
| `_get_wheel_air_spin_reward` | 仅对无接触轮返回轮速平方惩罚 |
| `_get_np3o_costs` | 计算 NP3O 约束成本(倾斜/高度/角速度/力矩/关节速度) |
| `_get_observations` | RL 钩子:完整观测拼接(见 1.2) |
| `get_privilaged_obs` | 返回特权观测(终止时写入 extras) |
| `_get_rear2_rear1_joint_limit_terms` | rear2 关节限位的软位置/力矩/速度惩罚 |

### 奖励计算

| 函数 | 作用 |
|---|---|
| `_get_rewards` | RL 钩子:全部奖励项计算与加权求和(见 1.2) |
| `_postprocess_reward_terms` | RL 钩子:状态机任务级奖励缩放(见 1.2) |

### 调试打印与观测后处理

| 函数 | 作用 |
|---|---|
| `_debug_print_observation_stats` | 定期打印观测统计定位 value 爆炸 |
| `_debug_print_reward_and_state_stats` | 定期打印奖励与关键状态统计 |
| `_print_tensor_stats` | 打印 max/mean/std/有限率 |
| `_tensor_max_abs` | 返回张量有限值的绝对值最大 |
| `_should_print_value_debug` | 判断是否超阈值触发 ValueDebug 打印 |
| `_apply_frame_mask` | 按缓存保留帧数遮蔽历史帧(最新 k 帧) |
| `_resample_frame_mask_num_keep` | reset 时采样一次保留帧数(回合内固定) |
| `_clip_obs_component` | 按组件名对观测限幅(标量或 [min,max]) |
| `_scale_obs_component` | 对选定观测分量做限幅后缩放 |
| `_get_scale_alias` | 从缩放配置读取别名键的缩放系数 |
| `_encode_joint_pos_obs` | 按 cfg 编码关节位置观测(如 sin/cos) |
| `_clip_scale_critic_component` | 对 critic 分量做限幅 + 缩放 |

### 特权观测支撑工具

| 函数 | 作用 |
|---|---|
| `_pad_flat_features` | 把特征张量补零到目标维度 |
| `_select_entity_features` | 按索引选择实体特征并整形 |
| `_resolve_name_patterns_to_indices` | 名称模式解析为实体索引 |
| `_get_privileged_extra_body_indices` | 解析特权附加观测的 body 索引 |
| `_get_privileged_extra_inertia_body_indices` | 解析惯性附加观测的 body 索引 |
| `_get_robot_physx_view` | 返回机器人 PhysX 根视图 |
| `_get_body_masses_tensor` | 读取各 body 质量张量 |
| `_get_body_mass_scale_tensor` | 读取质量缩放张量 |
| `_get_delay_time_lag_obs` | 输出当前电机/IMU 延迟 lag 观测 |
| `_get_body_inertias_tensor` | 读取 body 惯量张量 |
| `_get_body_material_tensor` | 读取 body 材料张量 |
| `_get_scan_dot_obs` | 由点扫描器输出离地高度扫描观测 |
| `_get_legacy_spring_state_obs` | 旧版弹簧状态观测(力/长度) |
| `_get_legacy_contact_force_obs` | 旧版接触力观测 |
| `_get_legacy_randomize_params_obs` | 旧版域随机化参数观测 |
| `_get_legacy_priv_latent_obs` | 旧版 priv_latent 观测拼接 |
| `_compute_uses_legacy_privileged_extra_obs` | 静态判断是否用旧版特权附加布局 |
| `_uses_legacy_privileged_extra_obs` | 返回缓存的旧版特权布局开关 |
| `_get_legacy_privileged_extra_obs` | 输出旧版特权附加观测 |
| `_get_critic_history_frame` | 取 critic 单帧历史(当前或按 cfg 堆叠) |

### 特权/成本观测与观测告警

| 函数 | 作用 |
|---|---|
| `_get_legacy_costs` | 旧版约束成本向量 |
| `_get_dynamic_priv_obs` | 拼装动态特权:关节刚度/阻尼/弹簧力等 |
| `_get_static_priv_obs` | 拼装静态特权:质量/惯量/材料/连杆 |
| `_get_privileged_extra_obs` | 按布局组合动态 + 静态特权附加观测 |
| `_debug_obs_alert` | 原始观测分量超阈值时打印细粒度告警 |
| `_per_env_max_abs_from_blocks` | 统计各 env 最大绝对值与非有限标志 |
| `_apply_termination_duration` | 终止条件需连续 N 步成立才真正终止 |
| `_clear_termination_duration_buffers` | reset 时清空终止持续计数器 |

### 结束判定与重置

| 函数 | 作用 |
|---|---|
| `_get_dones` | RL 钩子:完整结束判定(见 1.2) |
| `_reset_idx` | RL 钩子:完整重置流程(见 1.2) |
| `_get_randomize_params` | 收集刚度/阻尼/惯量/质量/材料等随机化参数 |
| `_range_pair_as_float` | 把区间对转 float 元组(带默认) |
| `_apply_root_state_uniform_vel_b` | 对根状态按体轴均匀采样位姿/速度 |
| `_get_reset_training_iteration` | 返回当前训练迭代数(课程用) |
| `_get_active_predefined_reset_air_modes` | 列出当前迭代激活的预定义空中重置模式 |
| `_clear_predefined_reset_air_command_limits` | 清除空中预定义重置的命令限幅 |
| `_parse_predefined_reset_air_limit_pair` | 解析空中重置命令限幅键值对 |
| `_set_predefined_reset_air_command_limits` | 设置空中预定义重置期间的命令限幅 |
| `_apply_predefined_reset_air_limit_pair` | 把限幅应用到单个速度/角速度分量 |
| `_apply_predefined_reset_air_command_limits` | 每步套用空中预定义重置命令限幅 |
| `_advance_predefined_reset_air_command_limit_timers` | 推进限幅计时器并到期解除 |
| `_apply_predefined_reset_air_height_limits` | 空中预定义模式覆盖高度命令限幅 |
| `_custom_reset_random` | RL 钩子:重置姿态/速度随机化(见 1.2) |

### 弹簧、命令刷新与收尾工具

| 函数 | 作用 |
|---|---|
| `_apply_spring` | 计算并写入弹簧(棱柱)关节 effort |
| `_resample_custom_cmd` | 命令计数器到期时重采样高度/权限/特殊模式 |
| `_inverse_kinematics` | 五连杆腿逆解:腿长 + 腿角 → 6 关节角 |
| `_on_command_updated` | RL 钩子:命令后处理全流程(见 1.2) |
| `_before_state_machine_command_updated` | 钩子:状态机更新前的命令覆盖(子类扩展点) |
| `_after_state_machine_command_updated` | 钩子:状态机更新后的最终命令覆盖(子类扩展点) |
| `_value_constrain` | 限幅目标值相对当前值的变化量 |
| `_init_episode_length` | 把秒数换算为控制步数(回合长度) |
| `_update_ground_height_estimate` | 用 height_scanner 射线估计地面高度(带缓存) |

---

# 二、wheelbipe_V13:地形命令层

## 2.1 这一层新增了什么

`class WheelbipeV13Env(Wheelbipe25V3Env)`。V13 在基座之上新增的核心是一套
**`TerrainCommandManager`(地形命令管理器)**:当机器人处于不同地形区块
(平地/坡道/台阶等)时,**按所在地形覆盖/禁用相应的速度、高度、偏航命令与训练机制**。
具体新增:

- 初始化 `TerrainCommandManager`,把地形区块与命令规则绑定;
- 重置与命令重采样时,按 env 当前地形同步命令覆盖、禁用特殊模式、按地形采样高度命令;
- 每步在状态机命令更新前后各做一次地形命令覆盖(切换地形区块时可强制重采样命令);
- play 模式头顶彩色球标记:颜色 = 所处地形类型;
- 对基座的若干掩码/采样方法做"地形感知"扩展(起跳禁用、预定义重置禁用、轴对齐朝向等)。

## 2.2 全部函数清单(20 个)

| 函数 | 作用 |
|---|---|
| `__init__` | 调父类初始化后:重找接触索引、创建 TerrainCommandManager、初始化地形命令状态并建地形标记 |
| `_is_terrain_type_marker_enabled` | 判断 play 模式是否显示地形类型标记(开关) |
| `_setup_terrain_type_marker` | 创建地形类型标记:每种地形一个按色相轮换颜色的球 |
| `_update_terrain_type_marker` | 每步刷新标记:各 env 头顶显示当前地形类型对应颜色 |
| `_get_terrain_command_manager` | 取地形命令管理器(未启用返回 None) |
| `_initialize_terrain_command_state` | 启动时按出生地初始化命令:轴对齐朝向、强制重采样、高度命令、命令覆盖 |
| `_sync_command_generator_command` | 把覆盖后的最终命令写回命令生成器,保持调试可视化一致 |
| `_disable_terrain_special_modes` | 按地形规则禁用某些 env 的特殊命令模式并同步命令 |
| `_restore_heading_closed_loop_yaw_command` | heading 类 env 保留命令生成器的闭环偏航率,不被地形覆盖 |
| `_update_axis_aligned_reset_heading_mask` | 重写:有地形管理器时由管理器给出轴对齐重置朝向掩码 |
| `_get_non_heading_axis_aligned_zero_mask` | 重写:被 `ang_vel_z_non_heading` 覆盖的 env 不清零偏航命令 |
| `_get_predefined_reset_air_disabled_mask` | 重写:叠加"该地形禁用预定义空中重置"的掩码 |
| `_get_predefined_reset_ground_disabled_mask` | 重写:叠加"该地形禁用预定义地面重置"的掩码 |
| `_get_jump_takeoff_disabled_mask` | 重写:叠加"该地形禁用起跳"的掩码 |
| `_sample_height_command` | 重写:高度命令改由地形管理器按 env 所在区块采样 |
| `_resample_custom_cmd` | 重写:命令重采样后同步地形覆盖并禁用特殊模式 |
| `_apply_terrain_command_overrides_for_current_step` | 每步核心:同步 env 所在区块,可选强制重采样,套用命令覆盖 |
| `_before_state_machine_command_updated` | 状态机命令更新前:先套用地形命令覆盖(允许切换时重采样) |
| `_after_state_machine_command_updated` | 状态机命令更新后:再套用一次覆盖(不重采样,做最终校正) |
| `_on_command_updated` | 命令后处理收尾:同步生成器命令缓存、刷新地形标记 |

---

# 三、wheelbipe_V14:云台 / 课程 / 录制层

## 3.1 这一层新增了什么

`class WheelbipeV14Env(WheelbipeV13Env)`。在 V13 之上新增四大块:

1. **云台(头部)控制**:pitch 固定角位置伺服;yaw 二选一——恒速自旋(速度目标)或
   航向锁定(自算 PD 力矩,需临时清零执行器位置增益);
2. **"小陀螺平移"特殊模式**:头部自旋的同时车身朝云台指向平移;目标速度/方向/身高在
   命令重采样时随机(多段区间),云台系速度投影到车体系写进 `self.command`;
   配套专属奖励项(9 个)与改写的 7 维控制模式观测,play 模式头顶绿球提示;
3. **粗糙地形高度课程学习**:训练按轮次推进课程等级,重置时把 env 出生点搬到对应难度
   地形(支持随机分配/毕业后全等级巩固),课程状态写进训练日志;越界判超时重置;
4. **速度轨迹录制**(play 用):按固定频率把某 agent 的速度/高度/各项奖励写 CSV,
   周期生成交互式 HTML 图表,退出时收尾。

## 3.2 模块级函数(4 个)

| 函数 | 作用 |
|---|---|
| `get_rough_height_offset_curriculum_cfg(cfg)` | 读"地形高度课程"参数并补齐默认值 |
| `get_rough_terrain_boundary_reset_cfg(cfg)` | 读"跑到地形边界就重置"参数并补齐默认值 |
| `get_training_progress_steps_per_iteration(cfg)` | 解析 PPO 每轮迭代包含的环境步数 |
| `get_extrapolated_training_iteration(env)` | 由 runner 锚点 + 累计步数外推当前训练轮次(也被 `commands.py` 外部导入使用) |

## 3.3 全部方法清单(57 个)

### 构造与训练进度

| 函数 | 作用 |
|---|---|
| `__init__` | 调父类后:初始化课程、找云台/腿/轮/接触索引、建云台状态张量、自检部件数量、准备轨迹录制 |
| `_is_gimbal_enabled` | 判断本任务是否启用云台(显式开关或按关节名推断) |
| `set_training_progress` | 公开回调:训练 runner 每轮把真实轮次/平均奖励同步给环境,供课程调度 |
| `_get_training_iteration` | 返回当前训练轮次(锚点 + 步数外推) |
| `_sync_command_generator_training_iteration` | 把推算轮次同步给命令生成器(特殊模式按轮次启停) |

### 粗糙地形高度课程

| 函数 | 作用 |
|---|---|
| `_rough_height_offset_curriculum_enabled` | 课程是否启用 |
| `_get_rough_height_offset_curriculum_iteration` | 课程推进的"时钟":当前训练轮次 |
| `_get_rough_height_offset_curriculum_level` | 计算当前课程等级:(等级号, 0~1 难度, 轮次) |
| `_apply_rough_height_offset_curriculum` | 课程核心:把 env 出生点搬到当前等级对应地形(支持随机分配) |
| `_append_rough_height_offset_curriculum_log` | 课程状态写入训练日志(TensorBoard 可见) |

### 速度轨迹录制(play 用)

| 函数 | 作用 |
|---|---|
| `_get_velocity_trace_cfg` | 读录制功能配置字典 |
| `_is_velocity_trace_enabled` | 录制功能开关 |
| `_ensure_velocity_trace` | 懒初始化:首次录制时建 CSV 文件、写表头(路径可加时间戳防重名) |
| `_close_velocity_trace` | 收尾(经 `atexit` 注册,进程退出时自动调):写最终 HTML、关文件 |
| `_get_velocity_trace_terrain_name` | 查询某 env 当前所在地形名(CSV 的 terrain 列) |
| `_select_velocity_trace_env` | 挑"录谁":优先配置指定,否则锁定首个符合条件的 env |
| `_record_velocity_trace` | 每个策略步按采样周期抓一帧(速度/高度/各项奖励)写 CSV |
| `_write_velocity_trace_html` | 把内存行数据渲染成交互式 HTML 并写盘 |

### 结束判定与重置钩子

| 函数 | 作用 |
|---|---|
| `_get_observations` | 重写:父类拼完观测后顺手录一帧轨迹数据 |
| `_reset_idx` | 重写:① 课程安排出生地形 → ② 父类通用重置 → ③ 重置云台 → ④ 课程写日志 |
| `_get_rough_terrain_boundary_time_out` | 检测哪些 env 跑出地形边界(布尔张量) |
| `_get_dones` | 重写:父类摔倒/超时判定之上,把"越界"并成超时(不计摔倒罚分) |
| `_custom_reset_random` | 重写:父类随机重置摆位后,再把云台也摆好 |

### 资产校验与工具

| 函数 | 作用 |
|---|---|
| `_resolve_names_to_indices` | 按名字顺序查索引,缺名直接报错(尽早发现资产/配置不匹配) |
| `_validate_v14_bookkeeping` | 启动自检:核对 V14 关节/连杆数量是否符合预期(腿 4、轮 2、弹簧 2、云台 2 等) |
| `_sample_uniform_range` | 工具:在 [low, high] 内均匀采 count 个数(区间反了自动纠正) |

### 云台:配置与启用判断

| 函数 | 作用 |
|---|---|
| `_get_gimbal_heading_control_cfg` | 读"云台航向锁定"配置 |
| `_is_gimbal_heading_control_enabled` | 航向锁定是否启用(开关 + 存在 yaw 关节) |
| `_get_gimbal_spin_translate_cfg` | 读"小陀螺平移"模式配置 |
| `_is_gimbal_spin_translate_enabled` | 小陀螺平移是否可用(开关 + 依赖航向锁定) |
| `_is_gimbal_spin_translate_marker_enabled` | 头顶提示球只在 play + 调试可视化时显示 |

### 云台:可视化标记

| 函数 | 作用 |
|---|---|
| `_create_gimbal_spin_translate_marker` | 创建标记物:激活=绿色发光球,未激活=隐形小黑球 |
| `_update_gimbal_spin_translate_marker` | 每步刷新:处于小陀螺平移模式的 env 头顶显示绿球 |

### 命令模式查询

| 函数 | 作用 |
|---|---|
| `_get_command_special_mode_mask` | 查询哪些 env 处于某个命名的"特殊命令模式"(布尔掩码) |
| `_get_gimbal_heading_control_mask` | 判断指定 env 这一步是否该用航向锁定控制 yaw |
| `_get_gimbal_spin_translate_mode_mask` | 哪些 env 当前处于小陀螺平移模式 |

### 云台:航向锁定 PD 控制

| 函数 | 作用 |
|---|---|
| `_ensure_gimbal_heading_pd_gain_tensors` | 保证航向 PD 的 kp/kd 张量存在且尺寸/设备正确 |
| `_get_gimbal_yaw_actuator` | 拿到 yaw 关节执行器对象(管刚度/阻尼) |
| `_capture_gimbal_yaw_actuator_gains` | 备份 yaw 执行器默认刚度/阻尼(PD 模式要临时清零) |
| `_set_gimbal_yaw_actuator_gains_for_heading_control` | 切换执行器增益:进 PD 清零,退出还原 |
| `_sample_gimbal_heading_targets` | 采样航向锁定目标:固定角 / 随机角 / 锁重置朝向 三种模式 |
| `_get_gimbal_yaw_link_heading_w` | 云台当前朝向:yaw 连杆世界系偏航角(无云台退回车体朝向) |
| `_get_gimbal_yaw_link_ang_vel_z_w` | 云台偏航角速度(PD 的 D 项用) |
| `_apply_gimbal_heading_pd` | 航向 PD:力矩 = kp×朝向误差 − kd×角速度,限幅后下发力矩目标 |

### 云台:恒速自旋

| 函数 | 作用 |
|---|---|
| `_sample_gimbal_yaw_velocity_targets` | 重采 yaw 恒速自旋的速度目标(区间内均匀随机,域随机化) |
| `_get_gimbal_yaw_joint_angle_wrapped` | 读 yaw 关节当前角度并规范到 [−π, π] |

### 小陀螺平移模式

| 函数 | 作用 |
|---|---|
| `_sample_gimbal_spin_translate_velocity` | 采样小陀螺平移目标:速度(支持多段区间)/方向/身高指令 |
| `_update_gimbal_spin_translate_samples` | 维护采样:刚进模式或命令重采时更新目标,退出模式时清零 |
| `_apply_gimbal_spin_translate_command` | 把云台系目标速度旋转到车体系写进 `self.command`(可覆盖身高指令) |
| `_get_gimbal_spin_translate_measured_lin_vel_yaw` | 实测车体速度反投影回云台系(指令投影的逆变换) |
| `_get_gimbal_spin_translate_reward_terms` | 小陀螺专属奖励项 9 个:追踪速度/速率/方向、罚超速/罚方向误差/罚站不住 |

### 云台:统一下发与重置

| 函数 | 作用 |
|---|---|
| `_apply_gimbal_targets` | 每步下发云台控制:pitch 位置伺服;yaw 按环境分拨走航向 PD 或恒速自旋 |
| `_reset_gimbal_joints` | 重置云台:直接写关节状态到仿真 + 重采目标 + 立即下发一次控制 |

### RL 主循环钩子(挂进框架调用链的入口)

| 函数 | 作用 |
|---|---|
| `_apply_action` | 重写:父类控制腿/轮 → 套用小陀螺命令 → 下发云台控制 → 刷新头顶提示球 |
| `_on_command_updated` | 重写:父类命令后处理 + 刷新小陀螺平移的命令投影 |
| `_postprocess_reward_terms` | 重写:小陀螺模式下作废普通速度追踪奖励项,加入小陀螺专属奖励项 |
| `_get_ctrl_mode_obs_raw` | 重写:小陀螺模式下改写 7 维控制模式观测(模式标志、指令速度、方向 sin/cos、云台角 sin/cos) |

---

# 四、三层分工速记

| 层 | 一句话职责 | 新增关键词 |
|---|---|---|
| wheelbipe25_v3 | "机器人怎么动、怎么算分":腿/轮/弹簧控制 + 全部奖励公式 + 观测/特权/延迟 + 重置随机化 | 基座、奖励库、状态机、起跳权限、Sim2Sim |
| wheelbipe_V13 | "在哪块地上就听哪块地的命令":TerrainCommandManager 按地形覆盖命令与训练机制 | 地形命令管理器、命令覆盖、地形标记 |
| wheelbipe_V14 | "加上会转的头和课程":云台两种控制、小陀螺平移模式、地形课程、轨迹录制 | 云台 PD/自旋、小陀螺平移、课程学习、CSV/HTML 录制 |

子类扩展基座的固定通道(全部走 `super()` 串联):`_pre_physics_step` / `_apply_action`(动作)、
`_get_observations` / `_get_ctrl_mode_obs_raw`(观测)、`_get_rewards` / `_postprocess_reward_terms`(奖励)、
`_get_dones`(结束)、`_reset_idx` / `_custom_reset_random`(重置)、
`_on_command_updated` / `_before/_after_state_machine_command_updated`(命令)。
