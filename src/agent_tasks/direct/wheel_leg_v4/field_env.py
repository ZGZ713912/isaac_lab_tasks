"""Wheel-leg V4 inference on the RMUC field, retaining its trained control ABI."""

import torch
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, quat_from_matrix

from agent_world.terrains.field_importer import FieldTerrainImporter
from agent_tasks.direct.deformable_suspension.field_surface import (
    FieldSurface, collision_coverage, resolve_field_asset, select_field_spawn,
)
from .env import WheelLegV4Env


class WheelLegFieldEnv(WheelLegV4Env):
    def __init__(self, cfg, render_mode=None, **kwargs):
        if not cfg.play or cfg.scene.num_envs != 1:
            raise ValueError("RMUC wheel-leg play requires a single V4 play environment")
        root, entry, digest = resolve_field_asset(cfg.field_root)
        self.field_package = dict(root=str(root), entry=str(entry), package_sha256=digest)
        cfg.terrain = cfg.terrain.copy()
        cfg.terrain.class_type = FieldTerrainImporter
        cfg.terrain.prim_path = "/World/Field"
        cfg.terrain.terrain_type = "usd"
        cfg.terrain.terrain_generator = None
        cfg.terrain.usd_path = str(entry)
        cfg.terrain.use_terrain_origins = False
        # On the training plane absolute z was also height above ground. Remove
        # only the field's world offset to preserve that physical height signal.
        cfg.use_absolute_height = False
        cfg.use_leg_length_as_height = False
        self._query_mesh_path = "/World/Field/HeightQueryMesh"
        for name in ("height_scanner", "right_wheel_height_scanner", "left_wheel_height_scanner", "dot_scanner"):
            scanner = getattr(cfg, name, None)
            if scanner is not None:
                scanner.mesh_prim_paths = [self._query_mesh_path]
                scanner.offset.pos = (0., 0., .025)
                scanner.debug_vis = False
        self.field_resets = 0
        self._field_spawn = None
        super().__init__(cfg, render_mode, **kwargs)
        wheel_ids = list(self._wheel_link_idx)
        root_position = self.robot.data.root_pos_w
        offsets = self.robot.data.body_link_pos_w[:, wheel_ids] - root_position[:, None]
        local = quat_apply_inverse(self.robot.data.root_quat_w[:, None].expand(-1, len(wheel_ids), -1), offsets)
        self._field_spawn = select_field_spawn(self.field_surface, local[0].cpu().numpy(), .06,
                                              cfg.field_spawn_xy, cfg.field_spawn_yaw)
        rotation = torch.tensor(self._field_spawn.pop("rotation"), dtype=torch.float32, device=self.device)
        self._field_spawn["quaternion"] = quat_from_matrix(rotation).tolist()
        origin = torch.tensor(self._field_spawn["origin"], device=self.device)
        self.terrain.env_origins[:] = origin
        self.scene.env_origins[:] = origin
        self._field_coverage = collision_coverage(root, self.terrain.field_path, self.field_surface.records)
        print(f"[field] wheel-leg V4; {len(self.field_surface.records)} colliders; "
              f"spawn={self._field_spawn['position']}", flush=True)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        self.cfg.terrain.num_envs = 1
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = FieldTerrainImporter(self.cfg.terrain)
        self.scene._terrain = self.terrain
        self.field_surface = FieldSurface.from_stage(self.scene.stage, self.terrain.field_path, self.device)
        # The installed RayCaster consumes one Mesh. Aggregate only enabled
        # collision triangles, without adding a second physical or visible floor.
        with Usd.EditContext(self.scene.stage, Usd.EditTarget(self.scene.stage.GetSessionLayer())):
            mesh = UsdGeom.Mesh.Define(self.scene.stage, self._query_mesh_path)
            mesh.CreatePointsAttr(self.field_surface.vertices.tolist())
            mesh.CreateFaceVertexCountsAttr([3] * len(self.field_surface.triangles))
            mesh.CreateFaceVertexIndicesAttr(self.field_surface.triangles.reshape(-1).tolist())
            mesh.CreateSubdivisionSchemeAttr("none")
            mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        for name in ("height_scanner", "right_wheel_height_scanner", "left_wheel_height_scanner", "dot_scanner"):
            config = getattr(self.cfg, name, None)
            enabled = name != "dot_scanner" or bool(getattr(self.cfg, "enable_scan_dot", False))
            sensor = RayCaster(config) if config is not None and enabled else None
            setattr(self, name, sensor)
            if sensor is not None:
                self.scene.sensors[name] = sensor
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.terrain.field_path])
        self._field_scene_count = sum(p.IsA(UsdPhysics.Scene) for p in self.scene.stage.Traverse())
        if self._field_scene_count != 1:
            raise ValueError(f"Expected one framework PhysicsScene, got {self._field_scene_count}")
        self._field_material_verified = all(str(UsdShade.MaterialBindingAPI(
            self.scene.stage.GetPrimAtPath(record["path"])).ComputeBoundMaterial("physics")[0].GetPath())
            == self.terrain.material_path for record in self.field_surface.records)
        if not self._field_material_verified:
            raise ValueError("Field colliders must retain the wheel-leg physics material")
        light = sim_utils.DomeLightCfg(intensity=1000., color=(.75, .75, .75))
        light.func("/World/Light", light)

    def _simple_root_reset(self, env_ids):
        if self._field_spawn is None:
            return super()._simple_root_reset(env_ids)
        ids = self._as_env_ids_tensor(env_ids)
        pose = torch.tensor(self._field_spawn["position"] + self._field_spawn["quaternion"], device=self.device)
        self.robot.write_root_pose_to_sim(pose.expand(len(ids), -1), env_ids=ids)
        self.robot.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=self.device), env_ids=ids)
        if hasattr(self, "_stand_ref_pos_w"):
            self._stand_ref_pos_w[ids] = pose[:2]
        self.field_resets += 1

    def _base_height(self):
        position = self.robot.data.root_pos_w
        height, _, valid = self.field_surface.support(position[:, None], self._sim_step_counter)
        self._field_base_supported = valid[:, 0]
        fallback = self.scene.env_origins[:, 2]
        return position[:, 2] - torch.where(valid[:, 0], height[:, 0], fallback)

    def field_audit(self):
        result = self.field_surface.audit()
        result.update(self.field_package, physics_scene_count=self._field_scene_count,
                      spawn=self._field_spawn, resets=self.field_resets, collision_coverage=self._field_coverage,
                      material_contract=dict(all_colliders_bound=self._field_material_verified,
                          friction_combine_mode=self.cfg.terrain.physics_material.friction_combine_mode),
                      query_mesh=self._query_mesh_path, query_mesh_has_collision=False,
                      height_reference="height above actual field support")
        return result
