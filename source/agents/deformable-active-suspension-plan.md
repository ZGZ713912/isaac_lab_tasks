# Deformable（四平行四边形变轮距全向轮）主动悬挂 RL 训练 — 实施规划

- 日期：2026-09-06
- 状态：planned
- 相关代码：
  - 资产：`source/agent_world/agent_world/assets/usd_files/deformable_infantry/`（新转换，2026-09-06 22:18）
  - 任务：`source/agent_tasks/agent_tasks/direct/deformable_suspension/`
  - 参考：`source/agent_tasks/agent_tasks/direct/wheelbipe/`（框架惯例）、`source/agent_world/agent_world/assets/wheelbipe_V14_2.py`（资产惯例）
  - 工具：`scripts/tools/convert_deformable_urdf.sh`、`scripts/rsl_rl/{train,play}.py`、`scripts/eval_checkpoint.py`、`scripts/view_robot.py`

---

## 1. 背景与目标

**机器人**：deformable（变形底盘），**完全不同于 wheelbipe**——
不是"双腿双轮轮腿"，而是 **4 个平行四边形变形机构 + 4 个全向轮 + 可变的轮距/车高**。
每个角：`base → leg（平行四边形主动边）→ wheel_set（平行四边形从动边）→ wheel（全向轮）`，
主动悬挂 = 实时控制 4 个角的平行四边形伸展角，让车体
① 跟踪高度指令、② 保持水平姿态、③ 四轮着地承重，
从而在粗糙地形上把车体与地面激励"隔离开"（悬挂功能）；平行四边形同步/差动伸缩还会改变轮距（变轮距），
影响静态稳定多边形，是控制难点。

**任务目标**：沿用本仓库 RL 框架（Isaac Sim 5.1 + Isaac Lab 2.3 DirectRLEnv + RSL-RL），
**补全并跑通 deformable 主动悬挂训练**：完善 `env.py` / `env_cfg.py` / 资产模块等必要文件，
按 Isaac Lab 项目规范编写；**sim-to-real 红线 = 与 RMCS `rmcs_rl` 实机部署合同逐项同构**
（obs 22 / act 4 / 100 Hz / kp=200 kd=4，见 §2.4）。

**现状一句话**：任务骨架（env/env_cfg/agents/注册）已在 `deformable_suspension/` 存在且基本可用，
但① 引用的资产模块 `agent_world/assets/deformable_suspension.py` **刚被删除**（git 工作区 D），
② 新转换的资产落在 `usd_files/deformable_infantry/`（新 URDF，关节角度约定已改），
③ 因此现在 `Robotics-Deformable-Suspension-*` **一 make 就 ImportError** —— 本规划的第一步就是把链路重新接上，再逐项完善。

---

## 2. 事实快照（2026-09-06 代码 + URDF 核查）

### 2.1 机器人结构与动力学（来自 `usd_files/deformable_infantry/urdf/deformable_infantry.urdf`）

- 拓扑：`base_link` + 4 × { `leg_N`, `wheel_set_N`, `wheel_N` }，共 12 DOF：

| 关节（×4 角） | 类型 | 连接 | 限位 | URDF effort | 说明 |
|---|---|---|---|---|---|
| `joint_leg_N` | revolute | base_link ↔ leg_N | 0.0873 ~ 1.3439 rad（5°~77°） | 10 | 平行四边形**主动边**（唯一驱动） |
| `joint_wheel_set_N` | revolute | leg_N ↔ wheel_set_N | 0.0873 ~ 1.3439 rad | 10 | 平行四边形**从动边**，几何上 θ_ws ≡ θ_leg（虚拟弹簧耦合） |
| `joint_wheel_N` | continuous | wheel_set_N ↔ wheel_N | 不限 | 40 | 全向轮，**当前零驱动**（自由滚动） |

- 质量：base 16.23 kg；leg 0.435 / wheel_set 1.292 / wheel 0.585 kg → 整车 ≈ **25.5 kg**。
- 轮子碰撞体 = **球体 r=0.077 m**（全向轮的仿真简化：球面接触天然支持任意方向滚动/侧滑）。
- 名义站姿：q = 1.3439 rad（77°，腿与水平面夹角；`bak_q77` 备份表明角度约定刚与部署统一，
  量角器实测 5°~77°）。车体半对角线 ±0.12921 m、平行四边形臂长 ~0.137 m。
- **变轮距**：平行四边形折叠 → 轮子相对车体沿"对角外伸/内收 + 升降"运动，轮距与车高几何耦合
  （耦合映射目前**未标定**，见 §6.1）。

### 2.2 资产现状（转换管线，与 wheelbipe V14_2 同款新格式）

```
usd_files/wheelbipeV14_2_1/            ← 已入库的参考样例（只提交 configuration/ + 根 .usd）
usd_files/deformable_infantry/         ← 新转换，355MB，尚未入库
├── deformable_infantry.usd            ← 根文件（入口）
├── configuration/{*_base,*_physics,*_robot,*_sensor}.usd
├── meshes/  urdf/  config.yaml        ← 转换器复制产物（冗余 355MB 大头）
```

- 旧资产模块 `assets/deformable_suspension.py`（74 行，git HEAD 仍有）引用的是
  `usd_files/deformable_suspension/deformable_suspension.usd` —— **该文件已不存在**，模块已删。
- `scripts/tools/convert_deformable_urdf.sh` 的输出路径仍写死 `usd_files/deformable_suspension/`
  （与现状错位，需修）。
- `config.yaml` 里 `asset_path/usd_dir` 指向旧位置 `assets/deformable_infantry/...`（过期，需改）。
- 仓库约定：wheelbipe V14_2 的 USD 是**提交进 git** 的（28MB）；deformable 目录因含 STL/URDF 副本达 355MB，
  入库前需决定剪裁策略（§5.1-D）。

### 2.3 现任务骨架（可继续沿用的部分）

`deformable_suspension/` 下已有：
- `env.py`（293 行）：完整 DirectRLEnv —— 关节按名解析、接触索引正则映射、手工腿 PD + 平四虚拟耦合
  弹簧（`set_joint_effort_target`）、policy/critic 双流观测、终止判定、重置（默认位姿/随机 yaw/
  随机高度指令）、episode_sums 日志。
- `env_cfg.py`（180 行）：`DeformableSuspensionBaseEnvCfg` + Flat/Rough/FlatPlay/RoughPlay 四个子类；
  含合同参数、观测缩放、奖励权重、sim（200Hz，decimation 2 → **100Hz 控制**）、场景、地形（Rough =
  金字塔坡 30% + 随机粗糙 70%）。
- `agents/rsl_rl_ppo_cfg.py`：PPO runner cfg（[256,128,64]，value_loss 4.0，风格与 wheelbipe 一致）。
- `__init__.py`：4 个 gym 注册（`Robotics-Deformable-Suspension{-Rough}{,-Play}-v0`）。

### 2.4 部署合同（sim-to-real 红线，勿动 policy 侧）

| 项 | 值 |
|---|---|
| obs（policy）22 | `cmd3(全零预留) \| height_cmd1 \| ang_vel3(×0.5) \| gravity3 \| leg_pos4(×1.0) \| leg_vel4(×0.1) \| act4` |
| obs（critic）26 | policy 22 + `lin_vel3 + base_height1(×5.0)`（asymmetric） |
| act 4 | 4 × 腿关节位置 PD 目标；`action_scale=0.25`，target 夹取 [0, 1.05] |
| 腿 PD | kp=200 / kd=4（部署 position_kp/kd），力矩限幅 ±40 N·m |
| 平四耦合 | 虚拟弹簧：`τ = k(θ_leg−θ_ws)+d(ω_leg−ω_ws)`，k=1000 d=10，加载于 leg(−)/ws(+) |
| 高度指令 | [0.05, 0.17] m，默认 0.132；观测缩放 height_scale=5.0 |
| 控制频率 | 100 Hz（sim dt 1/200，decimation 2）== 部署 rl_inference_frequency |
| episode | 20 s；终止：base 触地>1N、姿态>35°、车高<0.03、NaN |

---

## 3. 参考 wheelbipe 什么、刻意不抄什么

**参考（框架/惯例层面）**：
1. 文件分工：`wheelbipe_V14/{env.py, env_cfg.py, cfg_utils.py}` + `agents/rsl_rl_ppo_cfg.py` + 包内
   `gym.register`；configclass 继承树组织（Base → Flat/Rough/Play 变体）。
2. env 钩子命名与职责（`_setup_scene/_pre_physics_step/_apply_action/_get_observations/_get_dones/
   _get_rewards/_reset_idx`），super() 串联的洋葱扩展通道（若将来分"基座/课程"两层）。
3. 观测拼接、`nan_to_num`、episode_sums + `extras["log"]` 的 TensorBoard 记录惯例。
4. 资产模块惯例（`ArticulationCfg`：`UsdFileCfg(usd_path=f"{AssetPath}/usd_files/...")` +
   `activate_contact_sensors/copy_from_source` + rigid/articulation props + `init_state` + actuators；
   见 `wheelbipe_V14_2.py`）。
5. 调试/验证工具链：`list_envs.py` / `view_robot.py` / `train.py` / `play.py --keyboard` /
   `eval_checkpoint.py` 全部任务无关，直接复用。
6. （可选，M3+）wheelbipe V14 的**课程**思想（`CurriculumCfgV14`、地形难度行、训练进度外推）
   与**轨迹录制**（CSV/HTML）可平移成"悬挂性能录制"。

**刻意不抄（deformable 结构性不同）**：
- ❌ 双腿双轮结构、五连杆腿逆解、轮速/腿位置双段动作切分、腿轮速度跟踪奖励族；
- ❌ 腾空-落地状态机（airborne/jump_takeoff/step_up/stair）——deformable 无跳跃需求；
- ❌ 云台 yaw/pitch、航向锁定、小陀螺平移模式——无云台；
- ❌ wheelbipe 那套"两条腿一条轮"特有的命令生成器（XY 速度+偏航+高度）。deformable 现在只有
  **高度指令**一个标量命令维度（cmd3 预留）。

---

## 4. 差距清单（按严重度）

| # | 差距 | 位置 | 影响 |
|---|---|---|---|
| G1 | 资产模块被删，env_cfg import 断裂 | `assets/deformable_suspension.py`(D) + `deformable_suspension/env_cfg.py:33` | 任务无法 make，**阻塞一切** |
| G2 | 新 USD 关节方向/限位约定已改（`bak_q77` diff：rpy/axis/limit 全调过） | 新 URDF | 旧 env 假设（默认 1.3439、夹取范围、耦合符号）需在新 USD 上**逐项复核** |
| G3 | USD 关节 effort limit（leg 10 N·m）< env 力矩限幅（40） | USD 转换产物 | `set_joint_effort_target` 可能被 PhysX 关节 effort 上限截断 → **PD 力不足、站不起来** |
| G4 | 转换脚本/产物路径错位、config.yaml 过期、355MB 冗余未剪裁、新 usd 未入库 | `scripts/tools/`、`usd_files/deformable_infantry/` | 无法复现、git 卫生差 |
| G5 | 高度↔腿角↔轮距几何映射未标定 | 无文件 | `default_height_cmd=0.132` 与 q=1.3439 的对应关系存疑（0.17>0.132 却 q 已是 77° 上限，语义需澄清） |
| G6 | Rough 地形难度行（10 行 × 0.4~1.0）**未接课程**，出生点固定 | env_cfg | 训练没有渐进难度 |
| G7 | cmd3 全零预留、无任何平移/扰动命令 | env | 主动悬挂只做"原地",可先接受，移动悬挂留 M4 |
| G8 | 无域随机化事件表（质量/摩擦/增益） | env_cfg | 泛化差，sim2real 缺口（wheelbipe 的 EventCfg 可移植） |
| G9 | 无 play 可视化辅助（接触/高度标记）与性能录制 | env | 调试体验差（可选 M3） |
| G10 | 四轮着地/力矩/接触的**接触体命名**依赖正则（`leg_.*|wheel_set_.*`），新 USD body 名需复核 | env | 静默漏配 → 奖励/终止错 |

---

## 5. 方案：文件级实施计划（Isaac Lab 规范）

### M0 — 资产接线（解 G1/G2/G3/G4）

1. **重建资产模块** `source/agent_world/agent_world/assets/deformable_infantry.py`
   （建议按机器人名命名，废弃 `deformable_suspension.py` 名；或保留原名只改路径——需确认命名统一）：
   ```python
   DeformableInfantryCFG = ArticulationCfg(
       spawn=sim_utils.UsdFileCfg(
           usd_path=f"{AssetPath}/usd_files/deformable_infantry/deformable_infantry.usd",
           activate_contact_sensors=True,
           copy_from_source=True,        # Isaac Lab 2.3 加载 Articulation 必需
           rigid_props=...               # 照 wheelbipe_V14_2.py 模板
           articulation_props=...        # fix_root_link=False 等
       ),
       init_state=ArticulationCfg.InitialStateCfg(
           pos=(0, 0, ~0.13~0.18),       # 以新 USD 名义站姿 smoke 实测为准（G5）
           joint_pos={"joint_leg_.*": 1.3439, "joint_wheel_set_.*": 1.3439,
                      "joint_wheel_.*": 0.0},
           joint_vel={".*": 0.0},
       ),
       actuators={
           "legs": ImplicitActuatorCfg(joint_names_expr=["joint_leg_.*"],
               stiffness=0.0, damping=0.0,
               effort_limit={"joint_leg_.*": 40.0}),          # ← 覆盖 USD 的 10（G3）
           "wheel_set": ImplicitActuatorCfg(..., effort_limit=...),  # 跟随耦合力矩上限
           "wheels": ImplicitActuatorCfg(joint_names_expr=["joint_wheel_.*"],
               stiffness=0.0, damping=0.0),                    # 零驱动
       },
   )
   ```
2. **同步** `deformable_suspension/env_cfg.py:33` 的 import 与 `robot_cfg` 引用。
3. **修复工具**：`convert_deformable_urdf.sh` OUT_DIR → `usd_files/deformable_infantry`；
   修正 `config.yaml` 路径；**剪裁入库内容**（对齐 wheelbipeV14_2_1：只提交根 .usd +
   configuration/，355MB 的 meshes/urdf 副本不入库或另放 `.gitignore`）。
4. **G2 复核清单**（新 USD 上跑 smoke 验证，见 M3）：关节顺序/名字、默认位姿 q=1.3439 是否仍在
   限位内、耦合符号方向（θ_ws 是否 == θ_leg 于名义位姿）、body 名（接触正则 G10）。

### M1 — env_cfg.py 完善（解 G6/G8，对齐 wheelbipe 组织）

- Base cfg 保持合同值（§2.4），**新增**：
  1. `EventCfg`（移植 wheelbipe `manager/mdp/isaaclab/events.py` 的随机化函数）：
     `add_base_mass ×[0.9,1.3]`、`add_leg/wheel_set/wheel_mass`、摩擦 ×[0.7,1.3]（材质层）、
     可选 PD 增益抖动（kp ×[0.9,1.1]）；`mode="startup"` 为主（低频慢变扰动）。
  2. 地形难度接入：Rough cfg 里把 10 难度行映射为**出生行 = f(训练进度)**（M2 中做），
     或首版简单化：`difficulty_range` 固定 + 出生随机行（先跑通）。
  3. `cfg_utils.py`（新建，仿 wheelbipe `cfg_utils.py` 职责）：放几何标定表
     `LEG_ANGLE_TO_HEIGHT`（lookup/多项式，G5）、名义站姿常量、课程默认参数、
     `get_*_cfg` 读取辅助。
  4. rewards 权重块保持 OrderedDict 并注释每项语义；高度 σ、接触阈值等继续可覆盖。
- Play cfg：高度指令改为**固定/正弦/键盘可调**（wheelbipe play 的 Z/X 调高度思路可借鉴）；
  episode 可缩短便于演示。

### M2 — env.py 完善（解 G5/G7/G10，逐钩子对照规范）

以现有 293 行骨架为底，**保留合同核心**（手工 PD + 耦合弹簧 + obs 拼接 + 终止 + 重置），补：

1. `__init__`：加 `_validate_deformable_bookkeeping()`（照 V14 `_validate_v14_bookkeeping`：
   腿 4/轮架 4/轮 4/DOF 12 自检，名字不符即报错——解 G10 静默漏配）。
2. `_get_observations`：policy 22 结构**一字不改**（红线）；critic 26 同。cmd3 保留占位；
   观测噪声/延迟如需加，只允许加在 **critic/额外组**（部署不可得信息不进 policy）。
3. `_get_rewards`：核对 12 项权重语义并**按 deformable 物理重审**：
   - `flat_orientation_x/y_exp`、`track_height_exp`、`four_wheel_contact`、`undesired_contact`
     （leg/wheel_set 不该触地）、`alive/termination` 保留；
   - 力矩/速度/加速度惩罚的系数按新质量（25.5kg）与 effort 上限复核；
   - **可选新增**：轮距变化率惩罚（防变轮距抖动）、左右/前后着地力差惩罚（防侧倾翘轮）、
     行程利用率项（把 q 拉回中位区间，防"全程顶到 77° 死区"）。
4. `_get_dones`：保留 base 触地/姿态/高度/NaN；**新增**：四轮中若 ≥1 轮**持续离地超 N 步**
   且非主动姿态调整期，可给 time_out 而非 terminated（防策略学会翘轮逃奖励）——需确认。
5. `_reset_idx`：关节默认位姿 + 轮子随机角 + yaw 随机 + calm start 保留；
   **高度指令采样**从均匀 [0.05,0.17] 改为**课程化**（M2 与地形难度联动）；
   初始 root 高度以 smoke 实测调（防初始穿透，G5）。
6. **记录**：`extras["log"]` 增加悬挂指标（高度误差 MAE、姿态角 RMS、四轮法向力均值/方差、
   q 均值/行程、episode 存活时长），供 tensorboard 判断训练质量。
7. （M3 可选）play 可视化：四轮接触点高亮/力箭头 + 头顶高度指令球（仿 wheelbipe play marker）。

### M3 — 验证与调通（先跑通，再谈质量）

1. **冒烟**：`view_robot.py --task=Robotics-Deformable-Suspension-Play-v0 --num_envs=1 --device=cpu`
   —— 确认新 USD 能加载、名义站姿站得住、耦合弹簧方向正确（G2/G3/G5 全在此暴露）。
2. **CPU 训练冒烟**：train.py 4 envs × 2 iterations（REPRODUCE 同款），确认 obs/act 维数、日志落盘。
3. **Flat GPU 训练**：`--task=Robotics-Deformable-Suspension-v0 --num_envs=1024~2048 --headless
   --device=cuda:0`，看是否学会跟踪高度/四轮着地/水平（1000 iter 内曲线应明显起色）。
4. **Rough 训练**：`-Rough-v0` 2048 envs 长训。
5. **评估**：`eval_checkpoint.py`（存活率/高度误差/姿态/四轮着地率）；导出 .pt/.onnx 走现有 exporter。
6. **Play**：`run_gui.sh python scripts/rsl_rl/play.py ... --keyboard`（Z/X 高度），肉眼验收。

### M4 —（后置，可另立任务）移动悬挂 / 变轮距专项 / 课程正式化

全向轮驱动（4 轮速度指令 → act 8 合同扩展，需部署侧同步）、行驶中隔振、轮距-高度解耦命令、
TerrainCommandManager 式地形命令覆盖、速度轨迹录制（悬挂 trace）。

---

## 6. 关键决策与建议默认值

| # | 问题 | 建议（默认） | 备注 |
|---|---|---|---|
| D1 | 资产模块命名 | 机器人名 `deformable_infantry.py`（+CFG 同名），任务目录/experiment 保留 `deformable_suspension` | 资产=机器人、任务=能力，命名分离更清晰 |
| D2 | 平四耦合实现 | 保持**虚拟弹簧**（k=1000/d=10，与部署同构），不换 rigid 约束 | 改刚性约束会偏离部署合同 |
| D3 | effort 上限 | 资产 cfg `effort_limit` 覆盖为部署标定值（legs 40 N·m），**不以 URDF 的 10 为准** | G3 |
| D4 | policy obs | 22 维**冻结**；新信息只进 critic/aux | 红线 |
| D5 | 高度-角度几何 | 首版用查表近似 `height ≈ h(q)`（smoke 实测 2~3 个 q 标定），合同里的 0.05~0.17 范围语义以部署文档为准 | G5，需实机侧确认 |
| D6 | Rough 课程 | M2 先随机出生行跑通；课程化列为 M4 | 渐进 |
| D7 | 域随机化 | M1 加入质量/摩擦/增益 EventCfg | wheelbipe 函数可直接 import |
| D8 | 入库策略 | 只提交 USD（configuration/ + 根 .usd）；meshes/urdf/config.yaml 看情况排除 | 对齐 wheelbipeV14_2_1 |

---

## 7. 任务清单

- [ ] **M0.1** 重建资产模块 `assets/deformable_infantry.py`（指向新 USD + effort_limit 覆盖）
- [ ] **M0.2** 修 `env_cfg.py` import；修 `convert_deformable_urdf.sh` 路径；改 config.yaml
- [ ] **M0.3** 剪裁并 `git add` USD 资产（configuration/ + 根文件）；确认 .gitignore 策略
- [ ] **M0.4** view_robot CPU 冒烟：新 USD 加载 + 名义站姿 + 耦合方向（G2/G3/G5 验收点）
- [ ] **M1.1** env_cfg：EventCfg 域随机化表 + cfg_utils.py 几何标定骨架 + 注释化奖励块
- [ ] **M2.1** env.py：`_validate_deformable_bookkeeping` 自检；接触体名/关节名复核（G10）
- [ ] **M2.2** env.py：奖励项重审与新增（轮距抖动/着地力差/行程利用率，按确认取舍）
- [ ] **M2.3** env.py：悬挂指标日志（高度误差/姿态 RMS/四轮力/行程）
- [ ] **M3.1** CPU 训练冒烟（4 envs × 2 iter）通过
- [ ] **M3.2** Flat 短训（~2k iter）曲线验收：高度跟踪 + 水平 + 四轮着地
- [ ] **M3.3** eval_checkpoint 定量评估 + play 肉眼验收
- [ ] **M3.4** Rough 长训启动（2048 envs, headless）
- [ ] **M4**（另立）课程正式化 / 移动悬挂 / 变轮距命令 / trace 录制

---

## 8. 开放问题（需人工确认）

1. **机器人名与任务名**：资产/任务统一叫 `deformable_infantry` 还是保留 `deformable_suspension`？
2. **G5 高度语义**：`default_height_cmd=0.132` / 范围 [0.05,0.17] 与 q=1.3439(77°) 的几何对应，
   实机或 CAD 有没有现成标定？新 URDF 的 q 是否 = "腿与水平夹角"（量角器约定）？
3. **URDF effort=10 vs env ±40**：部署电机实际力矩上限/torque_max 标定值是多少？（D3 依据）
4. **首版要不要 cmd3 平移/扰动**：主动悬挂"原地"验收即可，还是需要 rough 地形上被动滚动
   （轮子自由滚 vs 低速扰动）？
5. **轮子接触模型**：球体碰撞 r=0.077 只是近似；实机全向轮（辊子）摩擦模型是否需要在
   material 层做各向异性/滚动阻力标定？现有 static/dynamic=1.0 的粗糙处理够不够？
6. 高度指令是否需要**正弦/阶跃轨迹**测试（wheelbipe 有特殊高度模式），还是均匀随机采样即可？

---

## 9. 任务定义 v2：任意基准高度 + 任意朝向小坡上的均力触地与车身水平

> 2026-09-06 增补（用户细化 + 三个决策已确认）。**本节为 v2 任务定义**，
> 取代此前"原地高度跟踪"假设（§2.4 保留作部署合同基线，差异见 §9.9）。
> 已确认决策：① 轮子由**外部底层速度环**按命令驱动，RL 只管 4 个悬挂电机（act4 不变）；
> ② 保留**基准高度/角度命令**，wheelbipe 式——**一个 policy 覆盖全部基准**；
> ③ 地形 ≤5° 起步，**课程渐进**到 ~10~15°。

### 9.1 目标与优先级

在**任意朝向的小坡度/不平地面**上（车体同时可平动 + 自旋，含小陀螺式旋转），主动悬挂必须：

1. **四轮均力触地**（最高优先级）：4 个轮全部着地且法向载荷均衡；
2. **车身尽量水平**（次优先级）：roll/pitch ≈ 0（世界系）。行程不足时"尽量"——由奖励梯度天然体现；
3. **跟踪基准高度/角度**（软目标）：悬挂在**任意基准设定**下都能工作（一个 policy，参考 wheelbipe
   的 height_cmd 全范围采样模式）；
4. 常规：不拖地（leg/wheel_set 触地惩罚）、力矩/能耗/抖动小、不摔倒。

### 9.2 传感器（部署可得）→ 观测设计

| 组 | 维 | 内容 | 部署来源 | 备注/缩放 |
|---|---|---|---|---|
| cmd | 3 | vx, vy, yaw_rate 命令 | 上位机 | 外部伺服执行，策略知悉运动意图（扰动） |
| baseline | 1 | 基准车体高度 h_cmd（备选：基准腿角 q_cmd） | 上位机 | ×5；全范围采样见 §9.4 |
| ang_vel | 3 | 车体角速度 body | IMU 陀螺 | ×0.5 |
| gravity | 3 | 投影重力（姿态/水平误差源） | IMU 姿态 | ×1 |
| leg_pos | 4 | 腿关节角（相对默认） | 电机编码器 | ×1.0 |
| leg_vel | 4 | 腿电机速度 | 电机 | ×0.1 |
| leg_torque | 4 | 腿电机力矩/电流 —— **每角载荷代理，均力判据的部署源** | 电机驱动（电流） | 缩放待标定（见 C3） |
| wheel_vel | 4 | 轮速（可选，看需要） | 轮编码器 | ×0.1 |
| act | 4 | 上一步动作 | — | ×1 |

policy ≈ 26 维（含轮速 30）。**critic（特权）**：+ lin_vel3、真实车高1、4×轮接触力（sim 特有）、
坡度/地面信息——训练时给"均力"提供真值；部署侧用 leg_torque 代理替代。
参考 wheelbipe：policy 只放部署可得量；传感器延迟/噪声建模、观测裁剪可选（照 wheelbipe obs 惯例，
M2/M3 加）。

### 9.3 动作与运动模型

- **act = 4 × 腿关节位置 PD 目标**（kp=200/kd=4，合同不变；动作处理沿用 §10 闭链设计：只驱动
  `joint_leg_*`，`joint_wheel_set_*` 走耦合力矩）。
- **轮子 = 外部速度伺服**：env 内每控制步按命令 (vx, vy, yaw) 经**四全向轮逆运动学**解 4 个轮速
  目标下发（小陀螺 = 大 yaw 分量；需标定轮子安装方向/驱动轴朝向矩阵，见 C2）。
  RL 不学轮子；轮速目标 = 扰动源 + 运动执行器。

### 9.4 基准高度命令（wheelbipe 模式）

- h_cmd 每次重置在范围（如 [0.05, 0.17]，具体与几何标定 §6.1 联动）内**均匀采样**进 obs；
- 奖励 = 车高相对 h_cmd 的软跟踪（σ 放宽 + 低权重），**不与均力/水平抢优先级**；
- 关键：**不做按 h_cmd 分段的 gate**——一个 policy 全范围（wheelbipe 同一高度指令机制；
  "阶跃/正弦特殊模式"可后置 M4）。
- 语义备注：h_cmd 是**基准**而非硬跟踪目标：坡上为保四轮着地+水平，车高允许偏离基准，
  偏离只在"可实现"范围内惩罚（参考 wheelbipe `vel_height_gate_*` 的 gating 思想，v2 先用低权重软项）。

### 9.5 奖励设计（按优先级）

令每角法向载荷 L_i（sim = 轮接触力；部署 = leg 电机力矩代理）、车高 h、基准 h_cmd：

| 优先级 | 项 | 公式（草案） | 权重初值 |
|---|---|---|---|
| 1 | all_contact 门控 | min_i L_i > ε 的平滑门控（或 L_i 下限 softplus） | 高 |
| 1 | contact_balance 均力 | −(max L − min L)/mean L 或 exp(−var/σ²) | 最高 |
| 2 | flat_orientation | exp(−pgb_x²/σ_p) + exp(−pgb_y²/σ_r) | 次高 |
| 3 | baseline_tracking | exp(−((h−h_cmd)²)/σ_h)，σ_h 放宽 | 中低 |
| 3 | stroke_margin 行程裕量 | 各角 q 远离限位（行程中点软奖励），防顶死、留裕量 | 中 |
| 4 | 常规惩罚 | torque² / action_rate / leg_vel² / 非轮部件触地 / base 触地(终止) | 低~中 |

- **不做** yaw/平动速度惩罚（允许小陀螺+平动）；
- 权重初值只作起点，**消融定稿**（C8）：均力 vs 水平 vs 基准三者的相对权重是调参主战场。

### 9.6 地形与课程

- 起步：坡度 ≤ **5°**（0.087 rad），混合：平地 + 缓坡（随机朝向由车辆穿越角覆盖）+
  可选小粗糙（noise 0.005~0.015）与浅坑；
- **课程**：难度行/等级随训练进度提升坡度到 ~10~15°（0.175~0.26 rad）——机制参考 wheelbipe
  V14 rough 课程（`CurriculumCfgV14`、难度行出生、进度外推 `set_training_progress`），
  deformable 版课程变量 = 坡度幅度（外加随机粗糙幅度微增）；
- **任意朝向**：出生 yaw 随机 + 命令剖面含自旋段 → 车体以任意角度上坡/过包；
- 越界重置/超时：照 wheelbipe V14（地形边界重置）。

### 9.7 命令剖面（运动扰动）

每 2~4 s 重采样 (vx, vy, yaw)（范围含：平动段、**小陀螺自旋段**（|yaw| 大 / v 小）、静止段），
由外部轮速伺服执行；episode 10~20 s。

### 9.8 终止 / 重置

- 终止：车体触地、|roll/pitch| 超限（坡上以"尽力水平"为目标，超限角可放宽）、NaN/数值异常（照现骨架）；
- 重置：leg/ws 名义角起步（闭链一致）、随机 yaw、随机出生地形难度、h_cmd 全范围采样、
  cmd 剖面重采样。

### 9.9 与现状代码的差异（保留/修改/新增）

| 现状 | v2 变化 | 涉及文件 |
|---|---|---|
| cmd3 全零预留 | 填入 vx/vy/yaw（外部伺服执行） | env.py obs + env_cfg |
| 轮子零驱动 | **外部速度伺服**（4 轮速目标，omni 逆运动学） | env.py `_apply_action`、env_cfg wheel servo 参数 |
| height_cmd 0.132 固定 | h_cmd 全范围采样 + 软跟踪奖励 | env.py reset/rewards |
| 奖励 12 项 | 按 §9.5 新表（均力/水平/基准/行程/常规） | env.py `_get_rewards`、env_cfg rewards |
| terrain 固定难度 | 坡度 ≤5° 起步 + 课程渐进 | env_cfg terrain + curriculum 钩子 |
| 无轮速 obs | wheel_vel4（可选）进 policy | obs 拼接 |
| —— | **leg_torque4 obs（载荷代理）** | obs 拼接（新增） |

### 9.10 新增待办（并入 §7 清单）

- [ ] **C1** obs 重排落地：维度确认 + 缩放表 + torque 语义
- [ ] **C2** 外部轮速伺服：omni 逆运动学矩阵标定（轮安装方向）+ 轮速下发方式（velocity target / env 内 PD），
      smoke 验证"按命令走、能自旋"
- [ ] **C3** leg_torque obs 语义：用指令力矩（PD−τ_cpl，贴部署电流）还是 applied_torque —— 标定后定
- [ ] **C4** 新奖励表接入 + episode_sums/tensorboard 日志项（均力误差、倾角、行程裕量）
- [ ] **C5** h_cmd 全范围单 policy 验证：不同基准下悬停/坡上曲线都收敛
- [ ] **C6** 地形课程接入（≤5°→15°，进度外推照 wheelbipe）
- [ ] **C7** 命令剖面（含自旋段）+ 随机朝向出生 + 越界重置
- [ ] **C8** 奖励消融：均力 vs 水平 vs 基准 权重
- [ ] **C9** 评估指标集：均力误差 (max−min)/mean、倾角 RMS、行程裕量、存活率（eval_checkpoint 扩展）

---

## 10. 平四闭链力学处理（核心设计，参考 wheelbipe 五连杆）

> 2026-09-06 增补。来源：用户说明（Isaac Lab 无法处理闭链）+ wheelbipe25_v3/env.py、
> wheelbipe_V14_2.py、V14 env_cfg 五连杆几何参数代码实读。

### 10.1 问题本质：树 vs 环

- Isaac Lab `Articulation` = **树**（每个 body 只有一个父关节），PhysX 不支持闭环/齿轮类关节约束。
- 真实平四：轮架（wheel_set）与 base 之间还有**第 4 根杆**把两者连起来 → 每角实际是 **1 DOF**；
- URDF 为开链化把闭环切断：`base → leg(joint_leg_N, 电机) → wheel_set(joint_wheel_set_N) → wheel_N`。
  于是 URDF 里每角出现 3 轴（leg/ws/wheel），扣除轮轴后仍有 2 个"假 DOF"——`joint_wheel_set`
  在树里是**欠约束自由铰**，物理上它必须跟随 `joint_leg`（平行四边形对边平行）。
- 不处理的后果：轮架乱晃、约束反力丢失 → 力矩/接触/奖励全失真，训练出来是假的。

### 10.2 wheelbipe 的解法（代码事实，照抄清单）

wheelbipe 每侧是**闭链五连杆 + 2 个电机 + 弹性弹簧**，URDF 开链化后有 6 个腿关节
（rear1/rear2/front1..front4）+ 弹簧棱柱关节。它的分工（`wheelbipe_V14_2.py` + `env.py`）：

| 关节组 | 仿真配置 | 职责 |
|---|---|---|
| `legs_act`（真电机，2/侧） | IdealPDActuatorCfg：stiffness 60/damping 2/effort 40/velocity 17 + armature(DM8009) | **位置伺服** `set_joint_position_target` |
| `legs_inact`（闭链被切断处的被动关节） | stiffness=0、damping=0.01、armature=1e-4 | 近自由，只提供数值阻尼，靠"几何一致 + 微阻尼"跟随 |
| `spring`（真实弹性件） | 物理 prismatic 关节 + `_apply_spring()` 以 **effort 力控**（力曲线 + 伸展/压缩非对称阻尼） | 真实弹簧力 |
| `wheel` | 速度/位置伺服 | 轮 |

关键机制：
1. **任务空间 → 关节空间解析逆解**：`_inverse_kinematics(leg_length, leg_angle)`，
   用 `links_length=[0.1134, 0.135, 0.210]` + `alpha_offset[6]`（安装角偏移）+ 余弦定理
   解闭链全关节角 `α[0..5]`（6 个角全算出来）；
2. **只命令真电机关节**（`legs_act` 位置目标），被动关节**不设目标**——约束不靠 PhysX、
   靠"控制 + 几何一致性"实现；
3. **逆解用于重置摆位**：`_custom_reset_random` 把腿摆到闭链一致构型，防初始内应力爆开；
4. 数值经验：solver position iterations 12、被动关节加 armature、力矩/速度限幅按电机标定。

> 一句话：**wheelbipe 方案 = 开链树 + 真电机位置伺服 + 解析逆解保构型 + 弹性件力控 + 重置摆位一致。**

### 10.3 deformable 的对应（每角 1 DOF 平四 → 比五连杆更简单）

与 wheelbipe 的差别：五连杆 2 电机 2 DOF，任务空间(腿长,腿角)需要解三角形；平四 **1 电机 1 DOF**，
约束方程就是 `θ_ws = f(θ_leg)`（对边平行；f 可能是同号/镜像/带偏移——按新 URDF 标定，见 M0.4），
**不需要 IK**。对应设计：

1. **关节分工**（与部署合同一致）：
   - `joint_leg_*`：唯一被驱动关节，手工位置 PD（kp=200 kd=4，`set_joint_effort_target`，与 rmcs_rl 同构）；
   - `joint_wheel_set_*`：**永不设位置目标**，只受力矩（耦合约束力）；
   - `joint_wheel_*`：零驱动自由滚动（全向轮）。
2. **约束实现（虚拟刚弹簧 = wheelbipe"被动跟随 + 弹性约束"的集中化）**：
   ```
   τ_cpl = k·(θ_leg − θ_ws) + d·(ω_leg − ω_ws)      # 先按同号写，标定后改 f(θ)
   τ_leg -= τ_cpl        # 电机净力矩 = PD − 耦合反力（= 实机电机力矩）
   τ_ws  += τ_cpl        # 轮架处约束反力
   ```
   这就是现骨架 `env.py:135-147` 的做法，方向正确；要按 §10.4 标定/加固的是 k/d 与 f(θ)。
3. **"力矩符合平四"的物理含义与验收**：
   - 电机净力矩 `τ_leg = PD − τ_cpl` 必须等于实机电机力矩（同指令下与 rmcs_rl 电流折算曲线对比）；
   - 轮载传递路径 `wheel → wheel_set → (τ_cpl) → leg → base` 与闭链一致 → 四轮接触力、
     力矩惩罚项才可信；
   - 稳态 `|θ_leg − θ_ws|` 必须足够小（目标 < ~1°），否则轮架"松" → 悬挂刚度假性偏低。
4. **数值红线**：
   - k 不是越大越好：200Hz 显式积分下过大 → 高频振荡/发散。k=1000/d=10 起步；
     若震 → 提 solver position iterations（8→12，资产 cfg 里改）或给 ws 加 armature，
     而不是继续加 k；
   - **禁止**把 `joint_wheel_set` 也设位置目标（两伺服互锁 → 高频斗力）；
   - 重置必须 leg/ws 同角起步（现骨架已做），杜绝初始耦合误差。
5. 若将来需要"更刚的轮架"，可选 wheelbipe 式双臂再开链（轮架另接 base 铰 + 汇聚端打断），
   改动大、首发不做。

### 10.4 待办与验收（并入 M0.4 / M3.1）

- [ ] 标定 `f(θ)`：新 USD 上 smoke 量 `joint_leg`/`joint_wheel_set` 的名义角与旋转方向
      （同号 / 镜像 / 偏移常量，进 cfg 作为 `ws_angle_offset` 之类参数）；
- [ ] 耦合公式按标定更新；k/d 独立成 cfg 组（`coupling_cfg`），附调参记录；
- [ ] **悬停测试**：固定 q 指令跑 2000 步，统计 `θ_leg−θ_ws` 误差 RMS、力矩峰值、无振荡；
- [ ] **阶跃测试**：单角目标 ±0.2 rad，检查四轮法向力重分配平滑、无爆力矩；
- [ ] （有实机数据时）同指令下 leg 电流/位置曲线与部署对比。

---

## 11. 历史

- 2026-09-06：初稿（核查：新 USD 转换产物、URDF 结构、旧资产模块删除、任务骨架、wheelbipe 惯例）。
- 2026-09-06：增补 §10 平四闭链力学设计（用户说明 Isaac Lab 无法处理闭链；实读 wheelbipe
  五连杆代码：legs_act 位置伺服 / legs_inact 被动近自由 / `_inverse_kinematics` 解析逆解 /
  `_apply_spring` 弹簧力控 / 重置摆位一致）。
- 2026-09-06：任务定义 v2（§9）：外部轮速伺服 / 任意基准高度单 policy（wheelbipe 模式）/
  均力触地>水平>基准 的优先级奖励 / ≤5° 起步课程渐进 / leg_torque 载荷代理 obs（决策来自用户答复）。
- 注意：本仓库文件可能被并发修改；执行前先 `git status` 复核（资产/任务目录正在迁移中）。
