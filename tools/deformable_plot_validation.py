"""Create a standalone figure from completed reference-only Isaac replay."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=json.loads(args.report.read_text())
    data=np.load(report['traces'],allow_pickle=False)
    variant=int(np.argmin(abs(data['taus']-.0046)))
    time=data['time'];mask=time>=5.
    fig,axes=plt.subplots(4,2,figsize=(13,10),sharex=True,layout='constrained')
    for j,name in enumerate(report['joint_order']):
        real_angle=np.rad2deg(data['reference_physical_angle'][:,j])
        sim_angle=np.rad2deg(data['predicted_physical_angle'][:,variant,j])
        real_current=data['reference_current'][:,j]
        sim_current=data['predicted_current'][:,variant,j]
        for col,real,sim in [(0,real_angle,sim_angle),(1,real_current,sim_current)]:
            ax=axes[j,col]
            ax.plot(time[mask],real[mask],color='#206A9E',linewidth=1.2,label='Recorded')
            ax.plot(time[mask],sim[mask],color='#C35D27',linewidth=1.1,linestyle='--',label='Isaac Sim')
            ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
            ax.set_xlim(5,30)
        rms=report['metrics'][variant]['angle_rmse_deg'][j]
        axes[j,0].set_ylabel(name.replace('_',' ').title()+'\nPhysical angle (deg)')
        axes[j,0].set_title(f'Angle RMS error: {rms:.2f} deg',loc='left',fontsize=10)
        axes[j,1].set_ylabel('Motor current (counts)')
    axes[0,0].legend(loc='lower right',ncols=2)
    axes[0,1].legend(loc='lower right',ncols=2)
    for ax in axes[-1]:ax.set_xlabel('Step-stage time (s)')
    fig.suptitle('Deformable Real2Sim: reference-only closed-loop replay\n'
                 '2 ms command / feedback, staggered by 1 ms; nominal current lag 4.6 ms',fontsize=14)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=160)
    fig.savefig(args.output.with_suffix('.pdf'))
    plt.close(fig)
    print(str(args.output.resolve()))


if __name__=='__main__':main()
