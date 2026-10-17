from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from agent_world.assets.wheel_leg_V4 import WheelLegV4_CFG
from agent_tasks.direct.wheel_leg_v3.wheel_leg_task.env_cfg import WheelLegV3FlatEnvCfg


@configclass
class WheelLegV4FlatEnvCfg(WheelLegV3FlatEnvCfg):
    """Full V5-style 46D frame x 5 history task on the existing Wheel_leg USD."""

    robot_cfg: ArticulationCfg = WheelLegV4_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    ).copy()

    # Closed-chain USD needs independent inspectable clones, not Fabric replication.
    scene = InteractiveSceneCfg(
        num_envs=4096, env_spacing=4.0, replicate_physics=True, clone_in_fabric=False
    )

    action_space = 6
    num_single_obs = 46
    num_obs_hist = 5
    observation_space = 230
    num_single_privileged_obs = 92
    num_privileged_obs_hist = 1
    state_space = 92

    # V5 action contract.
    v4_action_clip: float = 3.0
    v4_wheel_action_clip: float = 7.5
    v4_leg_position_scale: float = 0.35
    v4_wheel_velocity_scale: float = 10.0
    v4_leg_kp: float = 60.0
    v4_leg_kd: float = 2.0
    v4_wheel_kd: float = 0.2
    v4_wheel_effort_limit: float = 3.837686567164179
    v4_wheel_joint_sign: tuple[float, float] = (1.0, -1.0)
    v4_policy_dt: float = 0.01

    # Full V5 task fields are present from the first version.
    v4_task_mode_count: int = 5
    v4_phase_count: int = 5
    v4_spring_compression_scale: float = 0.08
    v4_spring_rate_scale: float = 1.2
    v4_wheel_contact_threshold: float = 2.0
    v4_closure_penalty: float = -10.0
    v4_spring_margin_penalty: float = -1.0
    v4_termination_penalty: float = -1.0

    use_wheel_vel_control = False
    gas_spring_enabled = True
    use_spring = False
    num_costs = 0

    # V4 builds its own 46D V5-style frame, so the legacy 7D ctrl_mode block
    # must NOT be appended by the inherited V3 __post_init__.
    ctrl_mode_obs_enabled = False
    jump_takeoff_extra_obs_enabled = False
    use_frame_stack = False


@configclass
class WheelLegV4FlatPlayEnvCfg(WheelLegV4FlatEnvCfg):
    scene = InteractiveSceneCfg(
        num_envs=1, env_spacing=4.0, replicate_physics=True, clone_in_fabric=False
    )
    play = True
