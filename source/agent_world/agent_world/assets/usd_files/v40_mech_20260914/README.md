# V40 纯底盘 机械结构包（URDF + STL）

## 目录

- **`urdf_V4.0/`** — SolidWorks 导出的完整 ROS package：`urdf/urdf_V4.0.urdf`
  （7 link、6 joint 的整机结构描述）+ `meshes/` 7 个原始 STL + package.xml/launch。
  `package://urdf_V4.0/meshes/...` 相对引用，**保持此目录名和结构**即可在任何
  工具中加载（ROS 直接放 workspace；MuJoCo/Isaac Sim 导入 URDF 时指到该目录）。
- **`parts_split/`** — 15 个零件独立 STL（机械结构本体拆分件，原始 CAD 坐标，
  同环境导入即复原完整装配）：每侧 crank/lower_coupler/thigh/rocker_3hole/
  upper_coupler/shank/wheel + base_link。C0–C4 由 link1.STL 三角形连通域拆分，
  面索引溯源完整（L 7122 / R 7072 面）。
- **`kinematics.json`** — 闭链孔轴坐标、源关节 frame、装配分支（两级四杆，
  销轴间隙 <1e-5 m）。
- **`mass_table.md`** — 质量表（整机 13.404 kg，URDF 级 + 均匀密度诊断值）及
  置信度提醒。

## 提醒

- 这是**运动学+几何**包：URDF 里的 inertial 为 SolidWorks 自动计算（base 占
  81% 需对照 BOM 确认），闭链在 URDF 中被拆为串行树（连杆 C0–C4 属 link1 刚体组）
  ——动力学等效性见 kinematics.json 与训练仓库审计。
- 网格为视觉级导出，不适合直接 3D 打印。
