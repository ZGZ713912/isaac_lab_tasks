# RMUC 灰色围栏场地第三人称驾驶

在仓库根目录运行：

```bash
./play_rmuc.sh deformable
```

统一启动器及 wheel-leg 的地图 play 见 [RMUC 场地 play](rmuc-field-play.md)。完整命令为：

```bash
./run_gui.sh /home/noir/miniconda3/envs/isaaclab/bin/python -B scripts/rsl_rl/play_deformable.py \
  --task Robotics-Deformable-Suspension-Guarded-History-MLP-Real2Sim-v4 \
  --checkpoint logs/rsl_rl/deformable_guarded_mlp5_v4/2026-10-14_12-40-18_training_long_05/model_999.pt \
  --field-root assets/terrains/RMUC2026_V2_gray_fence_revision_03 \
  --camera-mode third-person --device cpu --num_envs 1 --render_hz 50
```

使用已验收的 Guarded MLP v4，checkpoint SHA256：
`4af3c4c4c75d7c1280b5cded94e50fb49bc7ece75125b4a664234361a1609dcf`。
这次增加场地和驾驶交互，没有重新训练或替换模型。

## 操作

先点击 Isaac Sim 的场地视口，使其接收驾驶输入。
驾驶激活后，WASD 和 Q/E 由车辆接管，不再触发编辑器的移动/旋转工具快捷键。
点击其他面板或输入框会停止驾驶输入；回到场地视口再点击一次即可继续。

如果旧窗口正在反复报 `AttributeError: 'str' object has no attribute 'name'`，
请退出该 play 窗口后重新运行 `./play_rmuc.sh deformable`。已启动的进程不会自动加载代码修复。
2026-10-16 的修复分别处理 Kit 的字符串字符事件和视口 UI 点击事件；只有按下/松开事件改变驾驶状态。

| 输入 | 行为 |
| --- | --- |
| W / S | 沿实际车头前进 / 后退，默认上限 0.8 m/s |
| A / D | 相对实际车头左移 / 右移，默认上限 0.5 m/s |
| 按住 Q / E | 连续向左 / 右调整车头目标朝向，默认 60°/仿真秒；可与 WASD 同时使用 |
| 松开 Q / E | 停止累加目标，车辆完成最后设定的朝向后保持 |
| 按住右键，鼠标左右移动 | 设置车头目标朝向，灵敏度 0.15°/像素；右移向右转 |
| 松开右键 | 释放鼠标，车辆继续转到最后的目标朝向 |
| 按住右键，鼠标上下移动 | 调整镜头俯仰，上移提高视角，限制 5–75° |
| 滚轮 | 缩放，距离限制 1–8 m |
| L | 清空平移指令和待完成转向 |
| R | 重置出生姿态、底盘驱动、观测历史和悬挂保护状态 |
| Esc / 窗口失焦 | 清空输入、取消转向并释放鼠标 |

组合键同时生效；相反方向互相抵消，斜向输入归一化。Q/E 同时按下不累加朝向。
第三人称中 X/Z 不再设置旋转速度。
转向使用环绕角误差，经车轮驱动实际车身，最大 1.5 rad/s，误差进入 1° 后停止旋转指令。
相机默认 3 m、俯仰 20°，跟随实际车头方向，并以世界竖直轴保持水平。
场地已有碰撞几何挡住相机时，会自动缩短镜头距离。

`play_deformable.py` 默认启用第三人称，W 始终沿相机水平朝向前进。
保留显式选择的普通 `--camera-mode follow`、`--camera-mode fixed` 和旧的 `--fixed_camera`；这些旧模式沿用原键位。
`--fixed_camera` 与第三人称同时指定会报错。可用 `--vx_max`、`--vy_max`、`--wz_max` 调整指令上限。

## 场地与出生点

入口从 `asset_package.json` 的 `entries.isaac_static_collision_usd` 读取。
使用资产原有米制、Z 向上和世界坐标，保留主地面的倾斜与负 Z 高度。
引用放在运行时 session layer，停用资产自带的 PhysicsScene，仅保留框架的一个 PhysicsScene。
原始场地文件不写回。

默认在主地面自动寻找四轮可支撑、车身四周及初始相机无遮挡的位置，
按四个轮胎支撑高度求根节点高度和地面切向初始姿态。
可追加 `--spawn-xy X Y --spawn-yaw DEG` 指定出生点与朝向；坐标使用资产原始世界米制，
朝向使用角度。指定点没有地面、四轮无法共面支撑或净空不足时明确报错。

静态查询汇总 376 个已启用碰撞 Mesh，共 19,356 个三角形，并按物理步缓存。
轮胎和悬挂净空查询同一套真实碰撞地形；无命中时标记该轮无支撑，并停止该轮牵引。
PhysX 处理法向接触，原有轮胎模型处理切向牵引。
场地继承原控制合同的物理材质与摩擦组合方式，保证轮胎的零 PhysX 摩擦保持有效。

资产碰撞覆盖仍是部分覆盖。来源清单中有 **687 个原始几何体实例没有已启用的碰撞体**，
包含未重建装饰物和禁用的薄覆层；验证 JSON 的 `field.collision_coverage.uncovered` 逐项列出路径与名称。
这些可见物体可能被车或镜头穿过。地面、坡面及围栏的查询验证不代表每件装饰物都经过碰撞验收。

## 时序与验证

CPU 单车，物理步长 1 ms，策略周期 5 ms（200 Hz 仿真时间），
五帧历史仍按 10 ms 间隔采样；GUI 按 50 Hz 仿真时间更新。
实际画面帧率取决于实时倍率，不能将上述频率解释为 200 Hz 的墙钟实时运行。

针对性测试：

```bash
/home/noir/miniconda3/envs/isaaclab/bin/python -B -m pytest -q -p no:cacheprovider \
  scripts/tools/test_deformable_field_play.py \
  scripts/tools/test_deformable_third_person_input.py \
  scripts/tools/test_deformable_play_performance.py \
  scripts/tools/test_play_lifecycle.py
```

在上述 play 命令后追加以下参数，执行 2,400 步有界 GUI 回放：

```bash
  --seed 123 --profile --debug_motion --max_steps 2400 \
  --teleop-replay outputs/deformable_rmuc_third_person_20261014/teleop_replay.json \
  --verify-input-api \
  --profile_output outputs/deformable_rmuc_third_person_20261014/replay_validation.json \
  --capture-output outputs/deformable_rmuc_third_person_20261014/replay_view.png
```

回放与实时输入使用相同状态处理路径。原生接口探针通过独立虚拟设备检查本机 Kit 事件、
视口边界、鼠标捕获和释放，并确认驾驶事件没有传播到后续视口导航处理器。
该探针仅在有步数上限的 GUI 回放中运行。

验证产物保存在 `outputs/deformable_rmuc_third_person_20261014/`。
`validation.json` 汇总时序、接触、目标角、朝向、历史重置、碰撞覆盖、资产哈希和进程退出结果。
先前失败的日志保留用于追踪问题，不计为通过记录。

2026-10-14 最终记录为 `replay_05.json` / `replay_05.log`：

| 项目 | 结果 |
| --- | --- |
| 有界 GUI 回放 | 2,400 策略步，12 s 仿真；退出码 0 |
| 实时倍率 / 渲染更新 | 0.209× / 10.46 帧每墙钟秒 |
| 碰撞场景 | 1 个 PhysicsScene；地面、坡面、围栏 PhysX 查询误差均小于 0.002 m |
| 接触与目标角 | 48 个记录点均为四轮接触；四腿目标角误差为 0° |
| 允许 2 s 回落后的实际角误差 | P95 0.115°，最大 0.134° |
| 正反 30° 目标转向 | 最终误差分别 0.973° / 0.995°；松开右键后仍完成转向 |
| W 相对车头平移 | 位移方向与车头偏差 0.019° |
| 历史重置 | 三次启动初始化和一次 R；历史/指令均清零，无额外环境重置 |
| 本机 Kit 原生输入 | 虚拟设备事件通过；后续导航收到驾驶事件数为 0，光标恢复成功 |
| 文件与进程 | 场地 910 个文件哈希一致，checkpoint 哈希一致；无 play/训练/退出监控残留 |
| 针对性测试 | 24 项通过 |

接触与角度统计每 0.25 s 仿真时间记录一次；不等于逐物理步统计。
最终驾驶轨迹没有遇到镜头遮挡；缩短镜头的分支经过独立几何测试，
早期 `smoke_01.json` 也记录了真实场地碰撞将 3 m 镜头缩短的情况。
原生输入通过程序注入验证接口与路由，没有对人手鼠标操作的手感做量化评估。

2026-10-16 增加 Q/E 键盘第三人称驾驶后，30 项回归测试通过，
并完成 `outputs/rmuc_field_install_20261016/deformable-keyboard-gui-01.json` 的 2,400 步回放。
170° 出生朝向下，Q 左转跨过 ±180°，E 反向转回；松键后的最终误差分别为 0.988° / 0.988°。
W 位移与实际车头偏差 0.005°，48 个采样点均四轮接触。
回放约 0.199×、9.96 次渲染更新/墙钟秒，进程退出码 0，结束后无仿真或训练进程残留。
完整安装及验证结果见 [统一场地说明](rmuc-field-play.md)。
