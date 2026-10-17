"""Export a traceable fitted candidate for explicit Real2Sim-v3 tasks."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fit',type=Path,required=True)
    p.add_argument('--isaac-replay',type=Path,required=True)
    p.add_argument('--current-fit',type=Path)
    p.add_argument('--closed-loop',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    fit=json.loads(args.fit.read_text()); replay=json.loads(args.isaac_replay.read_text())
    if not replay['all_finite'] or replay['future_state_feedback']:
        p.error('export requires finite open-loop Isaac replay')
    if Path(replay['fit']).resolve()!=args.fit.resolve():
        p.error('Isaac replay references a different fit')
    digest=hashlib.sha256(args.fit.read_bytes()).hexdigest()
    if replay.get('fit_sha256')!=digest:
        p.error('Isaac replay must record the SHA256 of this exact fit')
    gains=[fit['models'][name]['torque_per_count'] for name in fit['joint_order']]
    model={'schema_version':2,'status':'conditional_fit_with_out_of_fit_diagnostics',
        'absolute_shaft_torque_calibrated':False,
        'simulation_effort_mapping':{'effort_unit':'N*m at the URDF actuated leg joint',
            'nominal_torque_per_current_raw_nm':sum(gains)/len(gains),'effort_limit_nm':300.,
            'motor_command_limit_current_raw':2048.,'passive_stop_shares_motor_limit':False},
        'fitted_mechanism':fit,
        'validation':{'blind_acceptance':False,'isaac_replay':replay,
            'fit_sha256':digest,
            'replay_sha256':hashlib.sha256(args.isaac_replay.read_bytes()).hexdigest(),
            'limitations':['CAD mass fixes effective torque scale; no shaft torque sensor',
                '1Hz/contact stress has larger error than slow/fast sweeps',
                'grounded checks use common-motion virtual work, not measured wheel loads',
                'wear and backlash are not separately identifiable',
                'not a trained or vehicle-validated policy']}}
    if args.current_fit:
        model['current_response_fit']=json.loads(args.current_fit.read_text())
        model['can_timing']={'controller_dt_s':.001,'command_period_steps':2,
            'feedback_period_steps':2,'feedback_delivery_phase_steps':1,
            'phase_evidence':'fresh-command rows see feedback about1.6ms old; alternate tick sees new packet',
            'pure_transport_delay_identified':False}
    if args.closed_loop:
        closed=json.loads(args.closed_loop.read_text())
        snapshot=args.closed_loop.with_suffix('.model.json')
        if not snapshot.is_file() or hashlib.sha256(snapshot.read_bytes()).hexdigest()!=closed['model_sha256']:
            p.error('closed-loop replay requires its exact model snapshot')
        if json.loads(snapshot.read_text())['fitted_mechanism']!=fit:
            p.error('closed-loop replay used a different fitted mechanism')
        if not closed['all_finite'] or closed['measured_state_feedback']:
            p.error('closed-loop replay must be finite and reference-only')
        model['validation']['closed_loop']=closed
        model['validation']['limitations'].append('upper-stop current relaxation remains mismatched, especially RF')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(model,indent=2)+'\n')
    print(str(args.output.resolve()),flush=True)


if __name__=='__main__':main()
