"""Plot the accepted fixed-budget FM trace; this is not a validation accuracy."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
EVIDENCE=ROOT/'reports/prefeval-k1-l0-l2-20260924/evidence/write-training'
fig,axes=plt.subplots(1,2,figsize=(11,4.4),sharey=True)
for ax,arm,color in zip(axes,['A','B'],['#cc6d26','#176f9f']):
    rows=[json.loads(line) for line in (EVIDENCE/f'writer/{arm}/write/optimization.jsonl').read_text().splitlines()]
    assert [r['step'] for r in rows]==list(range(1,2049))
    values=np.array([np.mean([d['mse'] for d in r['draws']]) for r in rows])
    ax.plot(range(1,2049),values,color=color,alpha=.2,linewidth=.6,label='Batch mean (4 draws)')
    ax.plot(range(64,2049),np.convolve(values,np.ones(64)/64,mode='valid'),color=color,
            linewidth=2,label='Trailing 64-update mean')
    ax.set(title=f'Arm {arm}',xlabel='Optimizer update',yscale='log',xlim=(1,2048))
    ax.grid(alpha=.2)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False,fontsize=8,loc='upper right')
axes[0].set_ylabel('Official FM velocity MSE (log scale)')
fig.suptitle('Shared Writer: 64 preferences, 2,048 updates per arm',fontsize=14)
fig.text(.5,.025,'Training error only. A/B target distributions differ; PNG readback determines memory performance.',
         ha='center',fontsize=9,color='#444444')
fig.tight_layout(rect=[0,.07,1,.93])
for suffix in ['png','svg']:
    fig.savefig(EVIDENCE/f'fm-write-training.{suffix}',dpi=180)
plt.close(fig)
