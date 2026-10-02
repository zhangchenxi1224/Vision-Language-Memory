"""Display whole-state query transfer and original-task application separately."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent;r=json.loads((p/'endpoint-verified.json').read_text())
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
colors=['#2864aa','#e69f00','#208779']
for j,(part,label) in enumerate([('training','Training query forms (3)'),('heldout','Held-out query forms (2)')]):
    pairs=[r['complete'][f'png_cpu/{part}/state/K{k}'] for k in (1,2,3,4)]
    x=np.arange(4)+(j-.5)*.34
    axes[0].bar(x,[a/b for a,b in pairs],.32,label=label,color=colors[j])
    for pos,(a,b) in zip(x,pairs):axes[0].text(pos,a/b+.025,f'{a}/{b}',ha='center',fontsize=10)
axes[0].set(xticks=np.arange(4),xticklabels=['1','2','3','4'],xlabel='Addressed slots, including cleared slots',
    ylabel='Complete states / all states',title='Same saved PNG: recovery across queries',ylim=(0,1.2))
axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.17),ncol=1,frameon=False)
for j,(condition,label) in enumerate([('blank','Blank'),('png','Teacher PNG'),('text','Complete text reference')]):
    pairs=[r['mcq_counts'][f'{condition}/{k}'] for k in ('K1','multi')]
    x=np.arange(2)+(j-1)*.25
    axes[1].bar(x,[a/b for a,b in pairs],.23,label=label,color=colors[j])
    for pos,(a,b) in zip(x,pairs):axes[1].text(pos,a/b+.025,f'{a}/{b}',ha='center',fontsize=10)
axes[1].set(xticks=np.arange(2),xticklabels=['1 slot','Multiple slots'],ylabel='Original MCQ accuracy',
    title='Apply each active preference to its original task',ylim=(0,1.2))
axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.17),ncol=2,frameon=False)
for ax in axes:
    ax.set_yticks([0,.25,.5,.75,1]);ax.set_yticklabels(['0%','25%','50%','75%','100%'])
fig.suptitle('Fixed offline teachers: strong training-query recovery, weak transfer and application',fontsize=14)
fig.savefig(p/'endpoint-diagnostic-results.png',dpi=160)
