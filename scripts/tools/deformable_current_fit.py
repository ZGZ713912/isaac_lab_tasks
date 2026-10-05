"""Effective current response at the CSV resolution; delay/lag are not separable."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import lfilter
from deformable_fit_data import JOINTS, load_recording


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,default=Path('/home/noir/Documents/workspace'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    records=[load_recording(path,args.output.parent/'cache') for path in sorted(args.workspace.glob('deformable*/*.csv'))]
    records=[r for r in records if r['status'].get('completed')]
    results={}
    timing={}
    for j,name in enumerate(JOINTS):
        sequences=[]
        phase_checks=[]
        for r in records:
            a=r['arrays']; tc=a['command_prepare_time_s'][:,j]; tf=a['feedback_time_s'][:,j]
            tc,tf=tc-tc[0],tf-tc[0]
            tg=np.arange(0,tc[-1],.001)
            u=np.interp(tg,tc,a['command_current_raw'][:,j])
            keep=np.isin(a['stage'],[3,7,8,10])&(tf>.2)
            age=a['host_time_s']-a['command_prepare_time_s'][:,j]
            servo=a['servo_current_raw'][:,j]
            rounded=np.sign(servo)*np.floor(abs(servo)+.5)
            difference=rounded.clip(-2048,2048)-a['command_current_raw'][:,j]
            feedback_offset=a['feedback_time_s'][:,j]-a['command_prepare_time_s'][:,j]
            phase={}
            for label,mask in [('fresh_command',keep&(age<.0005)),('alternate_control_tick',keep&(age>=.0005))]:
                phase[label]={'rows':int(mask.sum()),'servo_vs_prepared_abs_p95_counts':float(np.quantile(abs(difference[mask]),.95)),
                    'feedback_minus_prepare_ms_p10_p50_p90':(1000*np.quantile(feedback_offset[mask],[.1,.5,.9])).tolist()}
            phase_checks.append({'file':r['path'],'phases':phase})
            sequences.append({'file':r['path'],'train':r['metadata']['condition']=='suspended' and '23-03-11' not in r['path'],
                'grid':tg,'u':u,'time':tf[keep],'y':a['current_raw'][keep,j],'stage':a['stage'][keep]})
        def prediction(params,s):
            delay,tau,gain,bias=params
            alpha=1-np.exp(-.001/max(tau,1.e-8))
            response=lfilter([alpha],[1,alpha-1],s['u'])
            return gain*np.interp(s['time']-delay,s['grid'],response)+bias
        def residual(params):
            return np.concatenate([prediction(params,s)-s['y'] for s in sequences if s['train']])
        fit=least_squares(residual,[.001,.001,1.,0.],bounds=([0,0,.8,-10],[.01,.01,1.2,10]),
                          loss='soft_l1',f_scale=5,x_scale='jac',max_nfev=100)
        checks=[]
        for s in sequences:
            error=prediction(fit.x,s)-s['y']
            checks.append({'file':s['file'],'used_for_fit':s['train'],
                'stage_metrics':{str(int(stage)):{'rmse_counts':float(np.sqrt(np.mean(error[s['stage']==stage]**2))),
                    'abs_p95_counts':float(np.quantile(abs(error[s['stage']==stage]),.95))} for stage in np.unique(s['stage'])}})
        results[name]={'effective_delay_s':float(fit.x[0]),'effective_lag_tau_s':float(fit.x[1]),
                      'gain':float(fit.x[2]),'bias_counts':float(fit.x[3]),'checks':checks}
        timing[name]=phase_checks
    document={'scope':'sampled_command_to_sampled_current_effective_response',
        'command_reconstruction':'linear interpolation of observed 5ms CSV samples; unrecorded 2ms packets unavailable',
        'delay_and_lag_independently_identified':False,
        'noise_note':'response residual includes aliasing and reconstruction error; it is not white sensor noise',
        'models':results,'command_feedback_phase_diagnostics':timing,
        'phase_model':'2ms command and 2ms feedback, feedback delivered on alternate 1ms controller tick',
        'source_files':[{'path':r['path'],'sha256':r['sha256']} for r in records]}
    args.output.write_text(json.dumps(document,indent=2)+'\n')
    print(json.dumps({name:{k:v for k,v in m.items() if k!='checks'} for name,m in results.items()},indent=2),flush=True)


if __name__=='__main__':main()
