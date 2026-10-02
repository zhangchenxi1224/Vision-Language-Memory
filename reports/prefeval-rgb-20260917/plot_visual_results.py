"""Publication-style view of verified measured results; no score computation."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main(path):
    root=Path(__file__).resolve().parent
    visual=json.loads(path.read_text(encoding='utf-8'))
    refs=json.loads((root/'references-v2-verified.json').read_text(encoding='utf-8'))
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    fig.patch.set_facecolor('white')
    colors=['#adb5bd','#246a73','#da8c42']
    x=np.arange(2)
    for i,(condition,label) in enumerate([('blank','Blank image'),('text','Explicit text reference'),('writer','Unchanged 4f PNG')]):
        pairs=[visual['baseline']['counts'][f'singleton/{split}/official_mcq'] if condition=='writer'
               else refs['counts'][f'{condition}/{split}/official_mcq'] for split in ('train','dev')]
        heights=[a/b for a,b in pairs]
        bars=axes[0].bar(x+(i-1)*.24,heights,.22,label=label,color=colors[i])
        for bar,(a,b) in zip(bars,pairs):axes[0].text(bar.get_x()+bar.get_width()/2,bar.get_height()+.025,f'{a}/{b}',ha='center',fontsize=9)
    axes[0].set(xticks=x,xticklabels=['Pilot train','Pilot development'],ylim=(0,1.15),ylabel='Strict MCQ accuracy',title='Same 96 original preference tasks')
    axes[0].legend(loc='upper center',bbox_to_anchor=(.5,-.13),ncol=1,frameon=False,fontsize=9)
    pairs=[visual['sentinel']['complete_state_counts']['K'+str(k)] for k in (1,2,3,4)]
    bars=axes[1].bar([1,2,3,4],[a/b for a,b in pairs],color='#246a73',width=.65)
    for bar,(a,b) in zip(bars,pairs):axes[1].text(bar.get_x()+bar.get_width()/2,bar.get_height()+.025,f'{a}/{b}',ha='center',fontsize=10)
    axes[1].scatter([1,2,4],[.9,.75,.75],marker='_',s=900,color='#a33636',label='Allocation threshold')
    axes[1].set(xticks=[1,2,3,4],ylim=(0,1.15),xlabel='Addressed slots, including cleared slots',ylabel='Complete PNG-state recovery',title='Fixed 40-state visual teacher sentinel')
    axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.13),frameon=False,fontsize=9)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.set_axisbelow(True);ax.grid(axis='y',alpha=.18)
    fig.suptitle('PrefEval RGB memory: frozen Reader and actual saved PNGs',fontsize=14)
    fig.savefig(root/'visual-results.png',dpi=200,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('verified_result',type=Path);a=p.parse_args();main(a.verified_result)
