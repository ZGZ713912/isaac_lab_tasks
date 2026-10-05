"""Check reduced URDF inertia/gravity against the loaded Isaac USD articulation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'source/agent_world')]
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app
try:
    import numpy as np
    import torch
    from isaaclab.sim import SimulationContext, SimulationCfg
    from isaaclab.assets import Articulation
    from agent_world.assets.deformable_V2 import DeformableInfantryCFG
    from deformable_urdf_dynamics import ReducedLegDynamics

    sim = SimulationContext(SimulationCfg(dt=.001, device=args.device))
    cfg = DeformableInfantryCFG.copy()
    cfg.prim_path = '/World/Robot'
    cfg.spawn.articulation_props.fix_root_link = True
    cfg.init_state.pos = (0., 0., 2.)
    robot = Articulation(cfg)
    sim.reset()
    sim._disable_app_control_on_stop_handle = True
    robot.update(.001)
    names = robot.joint_names
    projection = np.zeros((len(names), 4))
    for j in range(4):
        for prefix, multiplier in [('joint_leg', 1), ('joint_wheel_set', 1), ('joint_upper_leg', -1)]:
            projection[names.index(f'{prefix}_{j+1}'), j] = multiplier
    records = []
    model = ReducedLegDynamics()
    for q_scalar in np.linspace(0., 1.05, 7):
        q = np.full((1, 4), q_scalar)
        joint = torch.tensor(q @ projection.T, dtype=torch.float32, device=args.device)
        robot.write_joint_state_to_sim(joint, torch.zeros_like(joint))
        sim.forward()
        robot.update(.001)
        view = robot.root_physx_view
        mass = view.get_generalized_mass_matrices()[0].cpu().numpy()
        gravity = view.get_gravity_compensation_forces()[0].cpu().numpy()
        reduced_mass = projection.T @ mass @ projection
        reduced_gravity = gravity @ projection
        analytic_mass, analytic_gravity, _, _ = model.quantities(q)
        records.append({'q':q_scalar, 'isaac_inertia':reduced_mass.tolist(),
                        'urdf_inertia':analytic_mass[0].tolist(),
                        'isaac_gravity_nm':reduced_gravity.tolist(),
                        'urdf_gravity_nm':analytic_gravity[0].tolist(),
                        'max_inertia_abs_error':float(np.max(np.abs(reduced_mass-analytic_mass[0]))),
                        'max_gravity_abs_error':float(np.max(np.abs(reduced_gravity-analytic_gravity[0])))})
    result = {'joint_order':names, 'fixed_base':robot.is_fixed_base, 'records':records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('URDF_PROBE',json.dumps({'max_inertia_error':max(r['max_inertia_abs_error'] for r in records),
                                 'max_gravity_error':max(r['max_gravity_abs_error'] for r in records)}),flush=True)
finally:
    app.close()
