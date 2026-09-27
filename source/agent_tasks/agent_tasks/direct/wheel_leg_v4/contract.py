"""Static V4 interface contract used by tests and deployment tooling."""

V4_POLICY_ORDER = (
    "L_joint1", "LL_joint1", "R_joint1", "RR_joint1", "L_joint3", "R_joint3"
)
V4_CONTROL_ORDER = (
    "L_joint1", "LL_joint1", "L_joint3", "R_joint1", "RR_joint1", "R_joint3"
)
V4_POLICY_TO_CONTROL = (0, 1, 4, 2, 3, 5)
V4_LEG_POLICY_IDS = (0, 1, 2, 3)
V4_WHEEL_POLICY_IDS = (4, 5)
V4_SINGLE_OBS_DIM = 46
V4_HISTORY_LENGTH = 5
V4_ACTOR_DIM = V4_SINGLE_OBS_DIM * V4_HISTORY_LENGTH


def validate_v4_shapes(policy_obs, critic_obs, actions, critic_dim=92):
    if policy_obs.shape[-1] != V4_ACTOR_DIM:
        raise ValueError(f"V4 actor observation must be {V4_ACTOR_DIM}D")
    if critic_obs.shape[-1] != critic_dim:
        raise ValueError(
            f"V4 critic observation must be {critic_dim}D, got {critic_obs.shape[-1]}"
        )
    if actions.shape[-1] != 6:
        raise ValueError("V4 action must be 6D")
