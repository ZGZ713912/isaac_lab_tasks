"""Single-vehicle play on a framework-owned RMUC field, with the original tire law."""

import math
import numpy as np
import torch

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.math import quat_from_matrix
import isaaclab.sim as sim_utils

from . import cfg_utils as du
from .dynamic_env import DeformableDynamicEnv
from .field_surface import FieldSurface, collision_coverage, resolve_field_asset


class FieldTerrainImporter(TerrainImporter):
    def import_usd(self, name, usd_path):
        self.field_path = self.cfg.prim_path + "/" + name
        stage = sim_utils.get_current_stage()
        if UsdGeom.GetStageUpAxis(stage) != "Z" or not math.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.):
            raise ValueError("Field destination stage must use metres and Z-up")
        with Usd.EditContext(stage, Usd.EditTarget(stage.GetSessionLayer())):
            field = UsdGeom.Xform.Define(stage, self.field_path).GetPrim()
            if not field.GetReferences().AddReference(usd_path):
                raise RuntimeError(f"Could not reference field: {usd_path}")
            for prim in list(Usd.PrimRange(field)):
                if prim.IsA(UsdPhysics.Scene):
                    prim.SetActive(False)
            # Keep the trained tire law as sole owner of wheel tangential force.
            # Inherit the original terrain's multiply combine mode so a zero-
            # friction wheel stays zero on these referenced USD colliders too.
            self.material_path = self.cfg.prim_path + "/PhysicsMaterial"
            material = self.cfg.physics_material.func(self.material_path, self.cfg.physics_material)
            UsdShade.MaterialBindingAPI.Apply(field).Bind(UsdShade.Material(material),
                bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose="physics")
        self.terrain_prim_paths.append(self.field_path)


class DeformableFieldEnv(DeformableDynamicEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        if not cfg.play or cfg.scene.num_envs != 1 or not cfg.runtime_guard_enabled:
            raise ValueError("RMUC field play requires a single Guarded MLP v4 play environment")
        root, entry, digest = resolve_field_asset(cfg.field_root)
        self.field_package = dict(root=str(root), entry=str(entry), package_sha256=digest)
        cfg.terrain = cfg.terrain.copy()
        cfg.terrain.class_type = FieldTerrainImporter
        cfg.terrain.prim_path = "/World/Field"
        cfg.terrain.terrain_type = "usd"
        cfg.terrain.terrain_generator = None
        cfg.terrain.usd_path = str(entry)
        cfg.terrain.use_terrain_origins = False
        self._field_generation = 0
        self._field_wheel_cache = None
        self._field_reference_ground = 0.
        self.field_resets = 0
        self.field_reset_checks = []
        super().__init__(cfg, render_mode, **kwargs)
        self._field_coverage = collision_coverage(root, self.terrain.field_path, self.field_surface.records)
        self._field_spawn = self._select_spawn(cfg.field_spawn_xy, cfg.field_spawn_yaw)
        self._field_reference_ground = self._field_spawn["ground_z"]
        self.scene.env_origins[:] = torch.tensor(self._field_spawn["origin"], device=self.device)
        print(f"[field] {len(self.field_surface.records)} collision meshes; "
              f"spawn={self._field_spawn['position']} yaw={cfg.field_spawn_yaw:.1f}deg", flush=True)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        self.cfg.terrain.num_envs = 1
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = FieldTerrainImporter(self.cfg.terrain)
        self.scene._terrain = self.terrain
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.terrain.field_path])
        self.field_surface = FieldSurface.from_stage(self.scene.stage, self.terrain.field_path, self.device)
        self._field_material_verified = all(str(UsdShade.MaterialBindingAPI(
            self.scene.stage.GetPrimAtPath(record["path"])).ComputeBoundMaterial("physics")[0].GetPath())
            == self.terrain.material_path for record in self.field_surface.records)
        if not self._field_material_verified:
            raise ValueError("Field colliders must inherit the existing tire-law material contract")
        self._field_scene_count = sum(prim.IsA(UsdPhysics.Scene) for prim in self.scene.stage.Traverse())
        if self._field_scene_count != 1:
            raise ValueError(f"Expected one framework PhysicsScene, got {self._field_scene_count}")
        light = sim_utils.DomeLightCfg(intensity=1000., color=(.75, .75, .75))
        light.func("/World/Light", light)

    def _field_tick(self):
        return (self._sim_step_counter, self.robot.data._sim_timestamp, self._field_generation)

    def _support(self, points):
        return self.field_surface.support(points, self._field_tick())

    def _ground_height(self, points):
        height, _, valid = self._support(points)
        # Keep diagnostic/critic tensors finite when support is absent. This
        # retained reference never supplies a normal force or tire traction.
        return torch.where(valid, height, height.new_full((), self._field_reference_ground))

    def _normals_at(self, points):
        return self._support(points)[1]

    def _wheel_geometry_w(self):
        tick = self._field_tick()
        if self._field_wheel_cache is None or self._field_wheel_cache[0] != tick:
            geometry = super()._wheel_geometry_w()
            heights, _, valid = self._support(geometry[1])
            self._field_wheel_valid = valid
            if valid.any():
                self._field_reference_ground = float(heights[valid].mean())
            self._field_wheel_cache = (tick, geometry)
        return self._field_wheel_cache[1]

    @property
    def wheel_normal_forces(self):
        forces = super().wheel_normal_forces
        return torch.where(self._field_wheel_valid, forces, torch.zeros_like(forces))

    @property
    def base_height(self):
        position = self.robot.data.root_pos_w
        return position[:, 2] - self._ground_height(position[:, None])[:, 0]

    def _select_spawn(self, requested, yaw_deg):
        floor = max(self.field_surface.records, key=lambda item: item["upward_area_m2"])
        bounds = np.asarray(floor["upward_bounds"])
        if requested is None:
            xy = np.array(np.meshgrid(np.arange(bounds[0, 0]+1., bounds[1, 0]-1., .5),
                                      np.arange(bounds[0, 1]+1., bounds[1, 1]-1., .5))).reshape(2, -1).T
            xy = xy[np.argsort(np.linalg.norm(xy - (-10., -3.), axis=1))]
        else:
            xy = np.asarray([requested], dtype=float)
        q = torch.full((1, 4), float(self.cfg.q_cmd_choices[0]), device=self.device)
        centers, _ = du.wheel_geometry(q)
        centers = centers[0].cpu().numpy()
        yaw = math.radians(yaw_deg)
        forward = np.array((math.cos(yaw), math.sin(yaw), 0.))
        high = float(self.field_surface.bounds[1, 2]+1.)
        for candidate in xy:
            probe = torch.tensor([[*candidate, high]], dtype=torch.float32, device=self.device)
            height, normals, valid = self.field_surface.support(probe, tick="spawn")
            if not bool(valid[0]) or not bounds[0, 2]-.02 <= float(height[0]) <= bounds[1, 2]+.02:
                continue
            normal = normals[0].cpu().numpy()
            if normal[2] < math.cos(math.radians(3.)):
                continue
            fwd = forward - normal * np.dot(forward, normal)
            fwd /= np.linalg.norm(fwd)
            rotation = np.column_stack((fwd, np.cross(normal, fwd), normal))
            rotated = centers @ rotation.T
            probes = np.c_[candidate + rotated[:, :2], np.full(4, high)]
            h, n, v = self.field_surface.support(torch.tensor(probes, dtype=torch.float32, device=self.device), "spawn")
            if not bool(v.all()) or bool((n[:, 2] < math.cos(math.radians(3.))).any()):
                continue
            z = h.cpu().numpy()-rotated[:, 2]+du.WHEEL_RADIUS/n[:, 2].cpu().numpy()
            if np.ptp(z) > .002:
                continue
            ground_z = float(height[0])
            position = np.r_[candidate, float(z.max()+.0002)]
            blocked = False
            clearance_radius = 2. if requested is None else .7
            for elevation in (.12, .3):
                origin = np.r_[candidate, ground_z+elevation]
                for angle in np.linspace(0., 2*math.pi, 8, endpoint=False):
                    if self.field_surface.segment(origin, (math.cos(angle), math.sin(angle), 0.), clearance_radius) is not None:
                        blocked = True
                        break
                if blocked:
                    break
            if blocked:
                continue
            if requested is None:
                rear = (-math.cos(yaw)*math.cos(math.radians(20.)),
                        -math.sin(yaw)*math.cos(math.radians(20.)), math.sin(math.radians(20.)))
                if self.field_surface.segment(position + (0., 0., .12), rear, 3.) is not None:
                    continue
            quaternion = quat_from_matrix(torch.tensor(rotation, dtype=torch.float32, device=self.device)).tolist()
            return dict(position=position.tolist(), quaternion=quaternion, ground_z=ground_z,
                        origin=[*candidate.tolist(), ground_z], normal=normal.tolist(),
                        wheel_height_spread_m=float(np.ptp(z)), auto_selected=requested is None)
        raise ValueError(f"No clear four-wheel flat spawn at {requested!r}; use another --spawn-xy")

    def _reset_idx(self, env_ids):
        self._field_generation += 1
        self._field_wheel_cache = None
        super()._reset_idx(env_ids)
        spawn = self._field_spawn
        pose = torch.tensor(spawn["position"] + spawn["quaternion"], device=self.device).expand(len(env_ids), -1)
        self.robot.write_root_pose_to_sim(pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=self.device), env_ids=env_ids)
        self._field_reference_ground = spawn["ground_z"]
        self._field_generation += 1
        self._field_wheel_cache = None
        self.field_resets += 1
        cleared = (not bool(self._history_valid[env_ids].any())
                   and not bool(self._history[env_ids].any())
                   and not bool(self._sensor_fifo[env_ids].any())
                   and not bool(self._drive_command[env_ids].any())
                   and not bool(self._causal_history.valid[env_ids].any())
                   and not bool(self._causal_history.frames[env_ids].any())
                   and not bool(self._causal_history.feedback[env_ids].any())
                   and not bool(self._runtime_guard.valid[env_ids].any()))
        if len(self.field_reset_checks) < 64:
            self.field_reset_checks.append(dict(reset=self.field_resets, step=self.common_step_counter,
                                                history_and_drive_cleared=cleared))

    def field_audit(self):
        report = self.field_surface.audit()
        report.update(self.field_package, physics_scene_count=self._field_scene_count,
                      spawn=self._field_spawn, resets=self.field_resets,
                      reset_checks=self.field_reset_checks,
                      collision_coverage=self._field_coverage,
                      material_contract=dict(all_colliders_bound=self._field_material_verified,
                          friction_combine_mode=self.cfg.terrain.physics_material.friction_combine_mode,
                          static_friction=self.cfg.terrain.physics_material.static_friction,
                          dynamic_friction=self.cfg.terrain.physics_material.dynamic_friction),
                      coverage="Existing enabled asset colliders; visual decorations may have no collider.")
        return report

    def verify_physics_queries(self):
        import omni.physx
        interface = omni.physx.get_physx_scene_query_interface()
        points = self.field_surface.vertices[self.field_surface.triangles]
        cross = np.cross(points[:, 1]-points[:, 0], points[:, 2]-points[:, 0])
        normals = cross / np.linalg.norm(cross, axis=-1, keepdims=True).clip(1.e-12)
        centroids = points.mean(1)
        grade = np.degrees(np.arccos(normals[:, 2].clip(-1.,1.)))
        selections = dict(floor=np.flatnonzero(grade < 3.), ramp=np.flatnonzero((grade > 14.) & (grade < 22.)))
        fence = next((r for r in self.field_surface.records if "PerimeterFenceCollision" in r["path"]), None)
        if fence is not None:
            selections["fence"] = np.arange(fence["first_triangle"], fence["first_triangle"]+fence["triangles"])
        results = []
        robot_xy = self.robot.data.root_link_pos_w[0, :2].cpu().numpy()
        for label, candidates in selections.items():
            for index in candidates:
                if np.linalg.norm(centroids[index,:2]-robot_xy) < 1.:
                    continue
                origin = centroids[index] + normals[index]*.05
                direction = -normals[index]
                query = interface.raycast_closest(tuple(float(v) for v in origin),
                    tuple(float(v) for v in direction), .15, bothSides=True)
                distance = self.field_surface.segment(origin, direction, .15)
                if query["hit"] and distance is not None and str(query["collision"]).startswith(self.terrain.field_path):
                    error = abs(float(query["distance"])-distance)
                    results.append(dict(surface=label, query_error_m=error, passed=error < .002,
                                        collider=str(query["collision"])))
                    break
            else:
                results.append(dict(surface=label, passed=False, error="No physics/query agreement sample"))
        outside = self.field_surface.bounds[1] + 10.
        _, _, valid = self.field_surface.support(torch.tensor(outside[None], device=self.device), "no-support")
        results.append(dict(surface="outside-field", passed=not bool(valid.any()), supported=bool(valid.any())))
        try:
            self._select_spawn(outside[:2], 0.)
        except ValueError as exc:
            results.append(dict(surface="invalid-spawn", passed=True, error=str(exc)))
        else:
            results.append(dict(surface="invalid-spawn", passed=False))
        self.field_physics_probes = results
        return results
