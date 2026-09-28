"""Reproduce the fixed-snapshot loss figures and per-update CSV; no training changes."""
import csv
from datetime import datetime, timedelta
import gzip
import hashlib
import json
from pathlib import Path
import sys

if len(sys.argv)>1:
    sys.path.insert(0,sys.argv[1])
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
import numpy as np

ROOT=Path(__file__).resolve().parent
meta=json.loads((ROOT/'source.json').read_text())
compressed=(ROOT/'optimization-committed.jsonl.gz').read_bytes()
assert hashlib.sha256(compressed).hexdigest()==meta['gzip_sha256']
raw=gzip.decompress(compressed)
assert hashlib.sha256(raw).hexdigest()==meta['raw_jsonl_sha256']
assert meta['committed_step'] == meta['observed_log_step'] == 93440
assert meta['uncommitted_tail_omitted'] == 0
records=[json.loads(line) for line in raw.splitlines()]
steps=np.asarray([r['step'] for r in records])
assert np.array_equal(steps,np.arange(1,meta['committed_step']+1))
micro=np.asarray([[d['mse'] for d in r['draws']] for r in records],dtype=np.float64)
assert micro.shape==(len(steps),4) and np.isfinite(micro).all() and (micro>=0).all()
loss=micro.mean(axis=1)
assert (loss>0).all()
window=1000
cumsum=np.r_[0.,np.cumsum(loss)]
smooth=np.full(len(loss),np.nan)
smooth[window-1:]=(cumsum[window:]-cumsum[:-window])/window
for end in [1000,23360,46720,70080,len(loss)]:
    assert np.isclose(smooth[end-1],loss[end-window:end].mean(),rtol=1e-11)
with (ROOT/'training-loss.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.writer(f,lineterminator='\n')
    writer.writerow(['step','mean_fm_mse','trailing_mean_1000','micro0_mse','micro1_mse','micro2_mse','micro3_mse','grad_norm','mean_sigma','v0_draws','v1_draws','phase'])
    for i,r in enumerate(records):
        writer.writerow([r['step'],float(loss[i]),'' if i<window-1 else float(smooth[i]),*micro[i].tolist(),r['grad_norm'],sum(d['sigma'] for d in r['draws'])/4,sum(d['initial_variant']==0 for d in r['draws']),sum(d['initial_variant']==1 for d in r['draws']),'original' if r['step']<=23360 else 'extension'])
blocks=[]
for start in range(0,len(loss),1000):
    stop=min(start+1000,len(loss))
    blocks.append(dict(first_step=start+1,last_step=stop,updates=stop-start,mean_mse=float(loss[start:stop].mean()),median_mse=float(np.median(loss[start:stop]))))
with (ROOT/'loss-1000-step-blocks.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(blocks[0]),lineterminator='\n')
    writer.writeheader()
    writer.writerows(blocks)
summary=dict(committed_step=int(steps[-1]),rows=len(records),rolling_window=window,rolling_direction='trailing; first 999 values omitted',first_1000_mean=float(loss[:1000].mean()),last_1000_mean=float(loss[-1000:].mean()),last_1000_at23360=float(loss[22360:23360].mean()),last_1000_at46720=float(loss[45720:46720].mean()),raw_min=float(loss.min()),raw_max=float(loss.max()),raw_quantiles={str(q):float(np.quantile(loss,q)) for q in [.01,.1,.5,.9,.99]},matplotlib=matplotlib.__version__,numpy=np.__version__)
summary['last_1000_at70080']=float(loss[69080:70080].mean())
summary['change_last1000_vs23360_percent']=100*(summary['last_1000_mean']/summary['last_1000_at23360']-1)

font=Path('C:/Windows/Fonts/msyh.ttc')
if not font.exists():
    raise RuntimeError('Provide a CJK font and update font path before reproducing the Chinese figure')
font_manager.fontManager.addfont(str(font))
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=str(font)).get_name(),'font.size':11,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.edgecolor':'#b7c2cf','xtick.color':'#45556b','ytick.color':'#45556b','text.color':'#14263d','axes.labelcolor':'#34445a','pdf.fonttype':42,'svg.fonttype':'path'})
fig,axes=plt.subplots(2,1,figsize=(13.8,9.4),gridspec_kw={'height_ratios':[1.12,1]},facecolor='#ffffff')
fig.subplots_adjust(left=.09,right=.965,top=.835,bottom=.12,hspace=.43)
capture=(datetime.fromisoformat(meta['captured_at_utc'].replace('Z','+00:00'))+timedelta(hours=8)).strftime('%Y-%m-%d %H:%M')
fig.text(.09,.955,'B730 训练 loss：固定 93,440 步完整轨迹',fontsize=21,fontweight='bold',ha='left')
fig.text(.09,.912,f"完整断点 1–{steps[-1]:,} 步  |  数据快照：{capture} 北京时间  |  effective batch = 4",fontsize=11,color='#5c6b7d')
blue='#235ad1'
teal='#158578'
ax=axes[0]
ax.plot(steps,loss,color='#a4afbf',lw=.35,alpha=.38,rasterized=True,label='每步原始 loss')
ax.plot(steps,smooth,color=blue,lw=2.0,label='1,000 步后向均值')
ax.set_yscale('log')
ax.set_xlim(1,steps[-1])
ax.set_ylabel('FM velocity MSE（对数坐标）')
ax.set_title('完整轨迹：展示全部原始波动',loc='left',pad=12,fontsize=13)
ax.axvspan(1,23360,facecolor='#cbd5e1',alpha=.14,zorder=-2)
ax.axvline(23360,color='#596579',ls='--',lw=1.2)
ax.axvline(46720,color=teal,ls='--',lw=1.2)
ax.axvline(70080,color=teal,ls='--',lw=1.2)
ax.text(23360,.97,' 23,360\n 续训起点',transform=ax.get_xaxis_transform(),va='top',fontsize=9,color='#475569',bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=2))
ax.text(46720,.97,' 46,720\n 固定评测点',transform=ax.get_xaxis_transform(),va='top',fontsize=9,color=teal,bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=2))
ax.text(70080,.97,' 70,080\n 固定评测点',transform=ax.get_xaxis_transform(),va='top',fontsize=9,color=teal,bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=2))
ax.text(93440,.97,'93,440\n最终终点 ',transform=ax.get_xaxis_transform(),ha='right',va='top',fontsize=9,color=blue,bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=2))
ax.legend(loc='lower left',frameon=True,framealpha=.94,edgecolor='none',ncol=2,fontsize=10)
ax.set_xlabel('绝对 optimizer step')

ax=axes[1]
keep=steps>=23360
ax.plot(steps[keep],smooth[keep],color=blue,lw=2.1)
ax.axvline(46720,color=teal,ls='--',lw=1.1)
ax.axvline(70080,color=teal,ls='--',lw=1.1)
ax.set_xlim(23360,steps[-1]+500)
lo,hi=np.nanmin(smooth[keep]),np.nanmax(smooth[keep])
pad=(hi-lo)*.19
ax.set_ylim(max(0,lo-pad),hi+pad)
ax.set_title('续训细节：固定 1,000 步后向均值（线性坐标）',loc='left',pad=12,fontsize=13)
ax.set_ylabel('平均 FM velocity MSE')
ax.set_xlabel('绝对 optimizer step')
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
for s,color,description,xytext,ha in [(23360,'#64748b','续训起点',(12,22),'left'),(46720,teal,'固定评测点',(8,22),'left'),(70080,teal,'固定评测点',(0,-32),'center'),(int(steps[-1]),blue,'最终完整断点',(-10,22),'right')]:
    value=smooth[s-1]
    ax.scatter([s],[value],s=36,color=color,zorder=5)
    ax.annotate(f'{description}\n{s:,} 步：{value:.4f}',xy=(s,value),xytext=xytext,textcoords='offset points',ha=ha,va='center',fontsize=9,color=color,bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=2))
for ax in axes:
    ax.grid(axis='y',which='major',color='#e2e8f0',lw=.65)
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    ax.tick_params(axis='both',labelsize=10)
    ax.set_axisbelow(True)
fig.text(.09,.055,'口径：每步 loss 为 4 个 micro-batch MSE 的均值；平滑曲线不是新的训练目标。',fontsize=10,color='#566477')
fig.text(.09,.028,'包含 1–93,440 步全部已提交更新，抢占重放不重复计数。Loss 不等于偏好读回准确率。',fontsize=9,color='#667587')
for ext in ['png','pdf','svg']:
    fig.savefig(ROOT/f'b730-training-loss.{ext}',dpi=190,facecolor='white')
svg=ROOT/'b730-training-loss.svg'
svg.write_bytes(('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n').encode('utf-8'))
plt.close(fig)
summary['artifacts']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.iterdir() if p.suffix in ['.png','.pdf','.svg','.csv']}
(ROOT/'plot-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
