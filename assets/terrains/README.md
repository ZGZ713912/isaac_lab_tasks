# 本地场地资产

RMUC 地图安装位置：`assets/terrains/RMUC2026_V2_gray_fence_revision_03/`。
2026-10-16 从 `/home/noir/Downloads/RMUC2026_V2_gray_fence_revision_03` 完整复制，原件保留。
包内 910 个文件共约 866 MiB，复制前后 SHA256 一致。

大型资产包通过 `.gitignore` 保留在本机；启动器和地图接入代码由 Git 管理。
如果换电脑或清理了资产，可在仓库根目录恢复完整目录：

```bash
mkdir -p assets/terrains
cp -a /home/noir/Downloads/RMUC2026_V2_gray_fence_revision_03 assets/terrains/
```

必须保留全部相对目录，不能只复制入口 USD。入口由包内 `asset_package.json` 的
`entries.isaac_static_collision_usd` 解析，当前为：

`full_field/fence/revision_03_gray/training_scene_gray_m_zup.usda`

使用方式及碰撞覆盖限制见 [场地 play](../../docs/rmuc-field-play.md)。
