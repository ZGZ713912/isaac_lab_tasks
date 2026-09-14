# V40 纯底盘 质量表

来源：URDF inertial（SolidWorks 计算）+ metrology 均匀密度诊断分配。

## 总表

| 零件 | 质量 kg | 来源 | 备注 |
|---|---|---|---|
| base_link | 10.800 | URDF | 占整机 81%，需确认含电机/紧固件与否 |
| L/R_C0 曲柄 | 0.0458 / 0.0459 | 均密度诊断* | |
| L/R_C1 下连杆 | 0.0322 / 0.0322 | 均密度诊断* | |
| L/R_C2 大腿 | 0.1823 / 0.1822 | 均密度诊断* | |
| L/R_C3 三孔摇杆 | 0.0459 / 0.0459 | 均密度诊断* | |
| L/R_C4 上连杆 | 0.0197 / 0.0197 | 均密度诊断* | |
| L/R_shank 小腿 | 0.450 / 0.450 | URDF | 等效密度 ~2045 kg/m³，量级合理 |
| L/R_wheel 轮 | 0.200 / 0.200 | URDF | 等效密度 ~2326 kg/m³（橡胶+毂混合合理） |
| **整机合计** | **13.404** | URDF | |

\* C0–C4 五件按"统一密度、归一到 URDF link1 总质量 0.326 kg"分配（每侧合计
精确等于 0.326）。实际材料若为 7075 铝（ρ≈2850）则诊断值系统性偏低约 15%。

## 单件惯量

每件的 `inertia_unit_density_kg_m2`（单位密度惯量张量）在
`isaac_wheeled_rl_train/reports/v40_linkage_repair_20260914/independent_geometry_mass/metrology.json`，
乘以真实材料密度即得惯量；合成校验（平行轴定理 vs 源 URDF link1）已通过。

## 置信度提醒

1. base_link 10.8 kg 占 81%——SolidWorks 装配体里电机/紧固件是否计入需对照 BOM
2. C0–C4 为均匀密度假设，单件分配非真实材料值
3. 大腿网格不封闭（watertight=false），体积为近似
