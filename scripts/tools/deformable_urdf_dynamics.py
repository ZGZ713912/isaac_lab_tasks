"""Reduced fixed-base dynamics of the actual URDF, including mimic links.

Independent coordinates are joint_leg_1..4. Wheels are held at zero spin for
the suspended identification experiment. These are simulator-model quantities,
not an assertion that the CAD masses equal the as-built vehicle masses.
"""
from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
URDF = ROOT / 'source/agent_world/agent_world/assets/usd_files/deformable_V2/urdf/deformable_V2.urdf'
URDF_JOINTS = tuple(f'joint_leg_{i}' for i in range(1, 5))
# +X forward, +Y left: derive the physical corner from root joint positions.
CSV_IN_URDF_ORDER = ('right_front', 'left_front', 'left_back', 'right_back')


def _origin(element):
    if element is None:
        return np.eye(3), np.zeros(3)
    xyz = np.fromstring(element.get('xyz', '0 0 0'), sep=' ')
    rpy = np.fromstring(element.get('rpy', '0 0 0'), sep=' ')
    return Rotation.from_euler('xyz', rpy).as_matrix(), xyz


class ReducedLegDynamics:
    def __init__(self, urdf=URDF):
        robot = ET.parse(urdf).getroot()
        self.links = {}
        for link in robot.findall('link'):
            inertial = link.find('inertial')
            rotation, com = _origin(inertial.find('origin'))
            inertia = inertial.find('inertia')
            values = {k: float(v) for k, v in inertia.attrib.items()}
            matrix = np.array([[values['ixx'], values['ixy'], values['ixz']],
                               [values['ixy'], values['iyy'], values['iyz']],
                               [values['ixz'], values['iyz'], values['izz']]])
            self.links[link.get('name')] = (float(inertial.find('mass').get('value')), com, rotation, matrix)
        self.joints = []
        for joint in robot.findall('joint'):
            name = joint.get('name')
            mimic = joint.find('mimic')
            coordinate = URDF_JOINTS.index(name) if name in URDF_JOINTS else None
            multiplier, offset = 1., 0.
            if mimic is not None:
                coordinate = URDF_JOINTS.index(mimic.get('joint'))
                multiplier, offset = float(mimic.get('multiplier', '1')), float(mimic.get('offset', '0'))
            rotation, position = _origin(joint.find('origin'))
            axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
            self.joints.append((joint.find('parent').get('link'), joint.find('child').get('link'),
                                rotation, position, axis, coordinate, multiplier, offset))

    def quantities(self, q, gravity=None):
        """Return M, gravity compensation, whole-model COM and wheel centers.

        Supports q shape (N,4). Gravity is expressed in the base frame.
        Jacobians include every ancestor mimic contribution before reduction.
        """
        q = np.atleast_2d(q)
        n = len(q)
        gravity = np.broadcast_to([0., 0., -9.81] if gravity is None else gravity, (n, 3))
        rotations = {'base_link': np.broadcast_to(np.eye(3), (n, 3, 3))}
        positions = {'base_link': np.zeros((n, 3))}
        ancestors = {'base_link': []}
        mass = np.zeros((n, 4, 4)); compensation = np.zeros((n, 4))
        weighted_com = np.zeros((n, 3)); total_mass = sum(x[0] for x in self.links.values())
        remaining = list(self.joints)
        while remaining:
            progressed = False
            for joint in remaining[:]:
                parent, child, r0, p0, axis, index, multiplier, offset = joint
                if parent not in rotations:
                    continue
                rp = rotations[parent] @ r0
                origin = positions[parent] + np.einsum('nij,j->ni', rotations[parent], p0)
                world_axis = np.einsum('nij,j->ni', rp, axis)
                angle = np.full(n, offset) if index is None else q[:, index] * multiplier + offset
                rotations[child] = rp @ Rotation.from_rotvec(angle[:, None] * axis).as_matrix()
                positions[child] = origin
                ancestors[child] = ancestors[parent] + ([] if index is None else [(origin, world_axis, index, multiplier)])
                remaining.remove(joint)
                progressed = True
            if not progressed:
                raise ValueError('URDF joint tree cannot be resolved')
        for name, (link_mass, local_com, ri, inertia) in self.links.items():
            rc = rotations[name] @ ri
            com = positions[name] + np.einsum('nij,j->ni', rotations[name], local_com)
            weighted_com += link_mass * com
            jv = np.zeros((n, 3, 4)); jw = np.zeros_like(jv)
            for origin, axis, index, multiplier in ancestors[name]:
                jv[:, :, index] += multiplier * np.cross(axis, com - origin)
                jw[:, :, index] += multiplier * axis
            world_inertia = rc @ inertia @ rc.transpose(0, 2, 1)
            mass += link_mass * jv.transpose(0, 2, 1) @ jv + jw.transpose(0, 2, 1) @ world_inertia @ jw
            compensation -= link_mass * np.einsum('nji,nj->ni', jv, gravity)
        wheels = np.stack([positions[f'wheel_{i}'] + np.einsum('nij,j->ni', rotations[f'wheel_{i}'],
                            np.array([-.0055, 0., 0.])) for i in range(1, 5)], axis=1)
        return mass, compensation, weighted_com / total_mass, wheels

    def inverse_dynamics(self, q, velocity, acceleration, gravity=None):
        mass, compensation, _, _ = self.quantities(q, gravity)
        eps = 1e-4
        # Independent legs and rigid mimic constraints make M diagonal. The
        # derivative is included instead of assuming a constant CAD inertia.
        derivative = (self.quantities(np.asarray(q) + eps)[0] - self.quantities(np.asarray(q) - eps)[0]) / (2 * eps)
        return np.einsum('nij,nj->ni', mass, acceleration) + .5 * np.diagonal(derivative, axis1=1, axis2=2) * velocity**2 + compensation
