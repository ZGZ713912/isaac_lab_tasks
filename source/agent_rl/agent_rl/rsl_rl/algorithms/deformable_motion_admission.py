"""Training-only reference admission; this never enters the exported actor."""

import math

import torch


def motion_preservation_factor(actor_obs, critic_obs, *, relaxed_fraction=.1,
                               parking_speed_limit=.1, moving_error_limit=.15,
                               encoder_tolerance=.1, support_min_load=20.):
    """Relax a reference only when supported wheels track but the body slides.

    A newly changed command, weak contact, or encoder slew retains full reference
    protection. The critic's true body velocity/normal forces are training
    privilege, and do not change the actor's H5/32D sensor input.
    """
    if (actor_obs.ndim != 2 or actor_obs.shape[1] != 160
            or critic_obs.shape != (len(actor_obs),40)):
        raise ValueError("Motion admission requires raw H5/160D actor and 40D critic contracts")
    values=(relaxed_fraction,parking_speed_limit,moving_error_limit,encoder_tolerance,support_min_load)
    if (not all(math.isfinite(value) for value in values) or not 0 <= relaxed_fraction < 1
            or min(values[1:]) <= 0):
        raise ValueError("Motion admission requires finite positive limits and a relaxed fraction below one")
    with torch.no_grad():
        frames=actor_obs.detach().reshape(-1,5,32)
        latest=frames[:,-1]
        critic=critic_obs.detach()
        command=latest[:,1:3]
        changed=(frames[:,:,1:4]-frames[:,-1:,1:4]).abs().amax(dim=(1,2)) > 1.e-6
        encoder_error=(latest[:,30:32]-command).norm(dim=-1)
        body_error=(critic[:,32:34]-command).norm(dim=-1)
        parked=command.norm(dim=-1) < 1.e-6
        limit=torch.where(parked,body_error.new_tensor(parking_speed_limit),
                           body_error.new_tensor(moving_error_limit))
        supported=(critic[:,36:40]/.02).amin(dim=-1) >= support_min_load
        finite=torch.isfinite(actor_obs).all(-1)&torch.isfinite(critic).all(-1)
        sliding=supported & ~changed & finite & (encoder_error <= encoder_tolerance) & (body_error > limit)
        # Nonfinite training inputs are an error, not a reason to weaken a
        # preservation constraint and continue learning from bad states.
        if not finite.all():
            raise ValueError("Motion admission received nonfinite training states")
        return torch.where(sliding,body_error.new_tensor(relaxed_fraction),torch.ones_like(body_error))
