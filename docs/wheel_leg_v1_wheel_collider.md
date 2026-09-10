# Wheel_leg_V1 轮子碰撞体：现在是什么、要不要换成解析圆柱

结论先行（2026-09-11 实测）：

1. **轮子的几何本来就是圆柱**，不是球也不是胶囊：轮体 STL（`L/R_link3.STL`）是
   半径 **0.0600 m**、宽 **0.0250 m** 的圆盘，绕自身轴是 90 段的分段圆
   （外圈 90 个离散角度，面片平均约 4.1°，弦高误差 ≈ **3.6 µm**）。
2. **但碰撞体不是解析圆柱**：URDF 里轮子的 `<collision>` 用的是 STL mesh，
   Isaac 的 URDF importer 只能把 mesh cook 成 convex hull
   （`urdf/urdf_V4.0.urdf` + `config.yaml: collider_type=convex_hull`），
   落盘为 `physics:approximation = "convexHull"` 的
   `PhysicsMeshCollisionAPI`，即"90 边多边形盘"。
3. **换成真正的解析圆柱收益很小**，因为 90 边的多边形盘与圆柱的径向差只有 3.6 µm
   （相对轮半径 6e-5）；而且解析圆柱在 PhysX 里是**线接触**（圆盘侧面与地面），
   属于窄相位里较难处理的接触类型，可能带来抖动/穿透类的新问题 ——
   这也是为什么 Isaac 的 URDF importer 提供 `replace_cylinders_with_capsules`
   这种会把圆角化（把线接触变成点接触）的开关。
4. 因此：**不要为了"轮子更圆"去换**。只有当实测确实怀疑轮-地接触有问题
   （滚动抖动、静止时轮子微跳、接触力尖峰、低速爬行）时，才值得做 A/B 验证。

## 1. 证据（怎么确认的）

```bash
# 轮体几何（STL 顶点统计）：r 最大 0.060000，z 跨度 0.025000，轴向 90 个离散角度
python3 - <<'EOF'
import struct, math
d=open('.../meshes/L_link3.STL','rb').read()
n=struct.unpack_from('<I',d,80)[0]
P=[struct.unpack_from('<12f',d,84+50*i)[3+3*k:6+3*k] for i in range(n) for k in range(3)]
print(max(math.hypot(p[0],p[1]) for p in P), max(p[2] for p in P)-min(p[2] for p in P))
EOF

# USD 侧（需要 pxr；系统中 /usr/bin/usdcat 缺 libboost_iostreams.so.1.92.0）
# scripts/tools/inspect_usd — 或用带 pxr 的 python 直接读：
#   /Wheel_leg_V1/L_link3/collisions  ->  Xform, instanceable, prepend references </colliders/L_link3>
#     L_link3/node_STL_BINARY_  apiSchemas=[PhysicsCollisionAPI, PhysicsMeshCollisionAPI]
#                               physics:approximation = convexHull
#       mesh (UsdGeom.Mesh)     9204 points, 3068 tris, extent z∈[0.00785, 0.03285]
```

全部 7 个 link（base_link、L/R_link1/2/3）都是 `convexHull`；整个 USD 里
**没有**任何 `UsdGeom.Cylinder/Capsule/Sphere` 碰撞体。

轮子以自身轴承中心为原点；轮轴方向的换算已验证：`L_joint3` / `R_joint3`
的 `localRot0` 都是纯 Z 轴旋转，把父坐标系 Z 映射到子坐标系 **local +Z**，
即"轮的旋转轴 = 轮 link 的局部 +Z"，与 `UsdGeom.Cylinder` 默认 axis=Z 一致
（这也是换解析圆柱时唯一需要小心的对齐问题）。

## 2. 如果要换成解析圆柱：已备好的变体资产

已加两个工具（不动现有管线，纯增量）：

| 文件 | 作用 |
| --- | --- |
| `scripts/tools/prepare_wheel_leg_v1_urdf.py --wheel-collision cylinder` | 从 SolidWorks 原始 URDF 生成 `urdf/urdf_V4.0_cylwheel.urdf`：轮子 `<collision>` 由 mesh 换成 `<cylinder radius="0.0600000032" length="0.0250000004"/>`，视觉/惯性/关节**一个字节都没动**（diff 只有两处 `<collision>`） |
| `scripts/tools/convert_wheel_leg_urdf_cylinder.sh` | 把该 URDF 转成独立资产目录 `usd_files/Wheel_leg_V1_cylwheel/`，并校验物理层里出现解析 `Cylinder` |

命令：

```bash
# 1) 生成 cylinder 变体 URDF（已执行，产物已落盘）
python scripts/tools/prepare_wheel_leg_v1_urdf.py --wheel-collision cylinder

# 2) 转 USD（需要 GPU + isaaclab，产物目录独立，不覆盖现有资产）
bash scripts/tools/convert_wheel_leg_urdf_cylinder.sh

# 3) A/B：把 source/agent_world/agent_world/assets/wheel_leg_V1.py 的 usd_path
#    指向 Wheel_leg_V1_cylwheel/Wheel_leg_V1_cylwheel.usd，跑同一个任务/同一超参，
#    对比 episode length、轮子接触力（play_wheel_material_debug）、轨迹抖动。
```

注意：`IsaacLab-v2.3.2/source/isaaclab/isaaclab/sim/converters/urdf_converter.py`
里只有 `collider_type ∈ {convex_hull, convex_decomposition}` 两种选择，
**解析圆柱只能靠 URDF 用 `<cylinder>` 图元**（importer 会保留为
`UsdGeom.Cylinder` + `PhysicsCollisionAPI`），或者转换后手工改 USD。
另外 IsaacLab 文档明确写了：圆柱/圆锥碰撞体对三角网格有专门的平滑接触支持
（`--/physics/collisionApproximateCylinders=true` 可关掉，用来换性能）。

## 3. 顺带发现的两个小不一致（与本题相关，建议单独修）

1. `wheel_radius` 常量不一致：`wheel_leg_base/env_cfg.py:259` 和
   `wheel_leg_terrain/env_cfg.py:471` 的起飞状态机里是 **0.05**，
   而 `wheel_leg_task/env_cfg.py:665` / `state_machines/airborne.py` 用的是 **0.06**，
   实际轮半径是 0.0600 m。0.05 会让"腿长 → 车高"换算差 1 cm。
2. URDF 的 `config.yaml` 写 `solver_position_iteration_count: 12 /
   solver_velocity_iteration_count: 6`，但转换产物 USD 里 bake 的是 `32 / 1`；
   运行时 `WheelLegV1_CFG.spawn.articulation_props`（12/6）会在 spawn 时覆盖 USD 值，
   所以最终生效是 12/6 —— 但要知道 config.yaml 里的数字并不等于运行时值。

## 4. 一句话给决策用

> 轮子已经是"圆柱"，只是碰撞体用了 90 边凸包近似（误差 3.6 µm）。
> 换解析圆柱属于"理论更干净、收益极小、且引入线接触风险"的改动；
> 变体资产与转换脚本已备好，**只有当你确实观察到轮-地接触异常时**再跑 A/B。
