"""Static collision geometry queries; original USD assets are never saved."""

from pathlib import Path
import hashlib
import json
import math

import numpy as np
import torch
import warp as wp


def resolve_field_asset(asset_root):
    root = Path(asset_root).expanduser().resolve()
    package = root / "asset_package.json"
    data = json.loads(package.read_text())
    if data.get("units") != "metres" or data.get("usd_up_axis") != "Z":
        raise ValueError("Field must use metres and Z-up coordinates")
    entry = (root / data["entries"]["isaac_static_collision_usd"]).resolve()
    if not entry.is_relative_to(root):
        raise ValueError("Field entry must be inside its asset package")
    if not entry.is_file():
        raise FileNotFoundError(entry)
    return root, entry, hashlib.sha256(package.read_bytes()).hexdigest()


def collision_coverage(asset_root, field_path, records):
    """List original visual body instances without an enabled field collider."""
    inventory = Path(asset_root) / "full_field/source_inventory.json"
    report = Path(asset_root) / "full_field/checks/collision_coverage_report.json"
    if not inventory.is_file():
        return dict(status="unknown", reason="No source body inventory in this package")
    source = json.loads(inventory.read_text())
    definitions = {item["body_id"]: item for item in source["body_definitions"]}
    enabled = {item["path"].split("/Collisions/", 1)[0] for item in records if "/Collisions/" in item["path"]}
    coverage = json.loads(report.read_text()) if report.is_file() else {}
    disabled = coverage.get("disabled_overlays", {}).get("names", [])
    rejected = coverage.get("rejected_source_geometry", {}).get("names", [])
    uncovered = []
    matched = 0
    for item in source["body_instances"]:
        path = field_path + "/" + item["body_instance_path"].split("/", 2)[2]
        if path in enabled:
            matched += 1
            continue
        definition = definitions[item["body_id"]]
        name = definition["owner_product_name"]
        reason = "disabled-overlay" if name in disabled else (
            "rejected-source-geometry" if name in rejected else "no-enabled-collider")
        uncovered.append(dict(path=path, body_id=item["body_id"], name=name, reason=reason))
    return dict(status="partial", source_inventory=str(inventory),
                source_inventory_sha256=hashlib.sha256(inventory.read_bytes()).hexdigest(),
                source_report=str(report) if report.is_file() else None,
                source_report_sha256=hashlib.sha256(report.read_bytes()).hexdigest() if report.is_file() else None,
                source_body_definitions=len(definitions), source_body_instances=len(source["body_instances"]),
                enabled_source_body_instances=matched, uncovered_body_instances=len(uncovered),
                uncovered=uncovered)


def select_field_spawn(surface, wheel_centers_b, wheel_radius, requested=None, yaw_deg=0.):
    """Fit a flat, unobstructed spawn to all wheel contacts in source-world coordinates."""
    floor = max(surface.records, key=lambda item: item["upward_area_m2"])
    bounds = np.asarray(floor["upward_bounds"])
    if requested is None:
        xy = np.array(np.meshgrid(np.arange(bounds[0, 0]+1., bounds[1, 0]-1., .5),
                                  np.arange(bounds[0, 1]+1., bounds[1, 1]-1., .5))).reshape(2, -1).T
        xy = xy[np.argsort(np.linalg.norm(xy - (-10., -3.), axis=1))]
    else:
        xy = np.asarray([requested], dtype=float)
    yaw = math.radians(yaw_deg)
    forward = np.array((math.cos(yaw), math.sin(yaw), 0.))
    high = float(surface.bounds[1, 2]+1.)
    centers = np.asarray(wheel_centers_b)
    for candidate in xy:
        probe = torch.tensor([[*candidate, high]], dtype=torch.float32, device=surface.device)
        height, normals, valid = surface.support(probe, tick="spawn")
        if not bool(valid[0]) or not bounds[0, 2]-.02 <= float(height[0]) <= bounds[1, 2]+.02:
            continue
        normal = normals[0].cpu().numpy()
        if normal[2] < math.cos(math.radians(3.)):
            continue
        fwd = forward - normal * np.dot(forward, normal)
        fwd /= np.linalg.norm(fwd)
        rotation = np.column_stack((fwd, np.cross(normal, fwd), normal))
        rotated = centers @ rotation.T
        probes = np.c_[candidate + rotated[:, :2], np.full(len(centers), high)]
        h, n, v = surface.support(torch.tensor(probes, dtype=torch.float32, device=surface.device), "spawn")
        if not bool(v.all()) or bool((n[:, 2] < math.cos(math.radians(3.))).any()):
            continue
        z = h.cpu().numpy()-rotated[:, 2]+wheel_radius/n[:, 2].cpu().numpy()
        if np.ptp(z) > .002:
            continue
        ground_z = float(height[0])
        position = np.r_[candidate, float(z.max()+.0002)]
        blocked = False
        for elevation in (.12, .3):
            origin = np.r_[candidate, ground_z+elevation]
            for angle in np.linspace(0., 2*math.pi, 8, endpoint=False):
                if surface.segment(origin, (math.cos(angle), math.sin(angle), 0.),
                                   2. if requested is None else .7) is not None:
                    blocked = True
                    break
            if blocked:
                break
        if blocked:
            continue
        if requested is None:
            rear = (-math.cos(yaw)*math.cos(math.radians(20.)),
                    -math.sin(yaw)*math.cos(math.radians(20.)), math.sin(math.radians(20.)))
            if surface.segment(position + (0., 0., .12), rear, 3.) is not None:
                continue
        return dict(position=position.tolist(), rotation=rotation.tolist(), ground_z=ground_z,
                    origin=[*candidate.tolist(), ground_z], normal=normal.tolist(),
                    wheel_height_spread_m=float(np.ptp(z)), auto_selected=requested is None)
    raise ValueError(f"No clear {len(centers)}-wheel flat spawn at {requested!r}; use another --spawn-xy")


@wp.kernel
def _support_rays(mesh_id: wp.uint64, starts: wp.array(dtype=wp.vec3),
                  positions: wp.array(dtype=wp.vec3), normals: wp.array(dtype=wp.vec3),
                  valid: wp.array(dtype=wp.int32), maximum: float):
    i = wp.tid()
    origin = starts[i] + wp.vec3(0., 0., .025)
    positions[i] = starts[i]
    normals[i] = wp.vec3(0., 0., 1.)
    valid[i] = 0
    # Ignore back faces and vertical walls as support; they remain physical colliders.
    for attempt in range(8):
        hit = wp.mesh_query_ray(mesh_id, origin, wp.vec3(0., 0., -1.), maximum)
        if not hit.result:
            break
        if hit.normal[2] > .1:
            positions[i] = origin + wp.vec3(0., 0., -hit.t)
            normals[i] = hit.normal
            valid[i] = 1
            break
        origin = origin + wp.vec3(0., 0., -hit.t - .001)


@wp.kernel
def _segment_ray(mesh_id: wp.uint64, origin: wp.vec3, direction: wp.vec3,
                 maximum: float, distance: wp.array(dtype=wp.float32)):
    hit = wp.mesh_query_ray(mesh_id, origin, direction, maximum)
    distance[0] = -1.
    if hit.result:
        distance[0] = hit.t


class FieldSurface:
    def __init__(self, vertices, triangles, device="cpu", records=None):
        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.triangles = np.asarray(triangles, dtype=np.int32)
        self.device = device
        wp.init()
        self.mesh = wp.Mesh(points=wp.array(self.vertices, dtype=wp.vec3, device=device),
                            indices=wp.array(self.triangles.reshape(-1), dtype=wp.int32, device=device))
        self.bounds = np.array((self.vertices.min(0), self.vertices.max(0)))
        self.records = records or []
        self.query_count = 0
        self.cache_hits = 0
        self._tick = None
        self._cache = {}
        self._segment_distance = wp.zeros(1, dtype=wp.float32, device=device)

    @classmethod
    def from_stage(cls, stage, path, device):
        from pxr import Usd, UsdGeom, UsdPhysics
        xforms = UsdGeom.XformCache()
        vertices, triangles, records = [], [], []
        offset = 0
        for prim in Usd.PrimRange(stage.GetPrimAtPath(path), Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                continue
            approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            if approximation not in (None, "none"):
                raise ValueError(f"Field query needs exact triangle colliders: {prim.GetPath()} ({approximation})")
            geom = UsdGeom.Mesh(prim)
            points = np.asarray(geom.GetPointsAttr().Get(), dtype=np.float64)
            transform = np.asarray(xforms.GetLocalToWorldTransform(prim))
            points = (np.c_[points, np.ones(len(points))] @ transform)[:, :3]
            counts = np.asarray(geom.GetFaceVertexCountsAttr().Get())
            indices = np.asarray(geom.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
            if np.any(counts != 3):
                raise ValueError(f"Field collider must be triangulated: {prim.GetPath()}")
            faces = indices.reshape(-1, 3)
            if geom.GetOrientationAttr().Get() == "leftHanded":
                faces = faces[:, (0, 2, 1)]
            if np.linalg.det(transform[:3, :3]) < 0:
                faces = faces[:, (0, 2, 1)]
            face_points = points[faces]
            cross = np.cross(face_points[:, 1]-face_points[:, 0], face_points[:, 2]-face_points[:, 0])
            areas = np.linalg.norm(cross, axis=-1)
            upward = cross[:, 2] > .9 * areas
            records.append(dict(path=str(prim.GetPath()), triangles=len(faces),
                                first_triangle=sum(record["triangles"] for record in records),
                                upward_area_m2=float(areas[upward].sum()/2),
                                upward_bounds=np.array((face_points[upward].reshape(-1, 3).min(0),
                                                        face_points[upward].reshape(-1, 3).max(0))).tolist()
                                if upward.any() else None))
            vertices.append(points)
            triangles.append(faces + offset)
            offset += len(points)
        if not vertices:
            raise ValueError("Field contains no enabled triangle collision meshes")
        return cls(np.concatenate(vertices), np.concatenate(triangles), device, records)

    def support(self, points, tick=None):
        if tick != self._tick:
            self._tick = tick
            self._cache.clear()
        key = (tuple(points.shape), points.detach().cpu().numpy().tobytes())
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        shape = points.shape
        starts = points.reshape(-1, 3).contiguous()
        hits = torch.empty_like(starts)
        normals = torch.empty_like(starts)
        valid = torch.empty(len(starts), device=self.device, dtype=torch.int32)
        wp.launch(_support_rays, dim=len(starts), inputs=[self.mesh.id, wp.from_torch(starts, dtype=wp.vec3),
                  wp.from_torch(hits, dtype=wp.vec3), wp.from_torch(normals, dtype=wp.vec3),
                  wp.from_torch(valid), float(self.bounds[1, 2]-self.bounds[0, 2]+10.)], device=self.device)
        result = (hits.reshape(shape)[..., 2], normals.reshape(shape), valid.reshape(shape[:-1]).bool())
        self.query_count += 1
        self._cache[key] = result
        return result

    def segment(self, origin, direction, maximum):
        wp.launch(_segment_ray, dim=1, inputs=[self.mesh.id, wp.vec3(*origin), wp.vec3(*direction),
                  float(maximum), self._segment_distance], device=self.device)
        distance = float(self._segment_distance.numpy()[0])
        return None if distance < 0. else distance

    def audit(self):
        return dict(collision_meshes=len(self.records), query_triangles=len(self.triangles),
                    world_bounds_m=self.bounds.tolist(), query_count=self.query_count,
                    cache_hits=self.cache_hits, collision_records=self.records)
