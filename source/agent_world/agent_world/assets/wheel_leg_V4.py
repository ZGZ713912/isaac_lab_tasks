# =============================================================================
# Wheel_leg_V4 asset configuration.
#
# V4 keeps the Wheel_leg_V2 USD and closed-chain geometry, but exposes only the
# six real motor joints to the environment. Motor efforts are computed by the
# V4 environment, so all articulation actuators are effort-only.
# =============================================================================

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from agent_world import AssetPath
from agent_world.assets.wheel_leg_V2 import (
    DEFAULT_SPAWN_HEIGHT,
    DM8009_ARMATURE,
    LEGS_ACT_JOINT_NAMES,
    PASSIVE_JOINT_NAMES,
    WHEEL_JOINT_NAMES,
)


WheelLegV4_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{AssetPath}/usd_files/Wheel_leg_V2/Wheel_leg_V2.usd",
        activate_contact_sensors=True,
        copy_from_source=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=False,
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=8,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, DEFAULT_SPAWN_HEIGHT),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs_act": ImplicitActuatorCfg(
            joint_names_expr=LEGS_ACT_JOINT_NAMES,
            stiffness=0.0,
            damping=0.0,
            effort_limit_sim=40.0,
            velocity_limit_sim=1.0e9,
            armature=DM8009_ARMATURE,
        ),
        "wheel": ImplicitActuatorCfg(
            joint_names_expr=WHEEL_JOINT_NAMES,
            stiffness=0.0,
            damping=0.0,
            effort_limit_sim=3.837686567164179,
            velocity_limit_sim=1.0e9,
            armature=0.0,
        ),
        "legs_inact": ImplicitActuatorCfg(
            joint_names_expr=PASSIVE_JOINT_NAMES,
            stiffness=0.0,
            damping=0.0,
            effort_limit_sim=0.0,
            velocity_limit_sim=1.0e9,
            armature=0.0,
        ),
    },
)
