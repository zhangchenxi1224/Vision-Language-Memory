import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent
d = json.loads((ROOT / 'projection.json').read_text(encoding='utf-8'))
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False,
                     'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
colors = ['#1678b8','#d65a31','#268868','#9b57ad','#aa8121','#cf4c7a','#4d65b4','#487b76']
all_points = np.array([p['points'] for p in d['paths']])[:, :, :2]
lim = np.max(np.abs(all_points)) * 1.2
fig, axes = plt.subplots(2, 4, figsize=(14.6, 7.9), constrained_layout=True)
rows = []
for p, ax, color in zip(d['paths'], axes.flat, colors):
    seed, points = p['seed'], np.array(p['points'])
    ax.plot(points[:,0],points[:,1],lw=2,color=color)
    ax.scatter(*points[0,:2],s=55,facecolor='white',edgecolor=color,zorder=4,label='起点')
    ax.scatter(*points[-1,:2],s=58,marker='D',color=color,zorder=5,label='256 步终点')
    for step in [8,16,32]:
        ax.scatter(*points[step,:2],s=16,color=color,zorder=3)
    ax.set(xlim=(-lim,lim),ylim=(-lim,lim),aspect='equal',xlabel='公共 PC1',ylabel='公共 PC2')
    ax.set_title(f"起点 {seed} → 终点 {seed}\n真实位移 {p['endpointRMSEFromStart']:.4f} · 第 {p['step95Path']} 步走完 95% 路程",fontsize=10)
    ax.grid(alpha=.16)
    rows.append({'seed':seed,'start_pc1':points[0,0],'start_pc2':points[0,1],
                 'final_pc1':points[-1,0],'final_pc2':points[-1,1],
                 'endpoint_rmse_from_start':p['endpointRMSEFromStart'],
                 'cumulative_path_length':p['pathLength'],'step_95pct_path':p['step95Path'],
                 'off_plane_rmse':p['endpointDistanceFrom2DPlane'],'final_mcq_correct':4,'final_mcq_views':4})
fig.suptitle('同一道选择题：8 条真实轨迹分别如何移动？\n同一投影平面、同一坐标尺度；空心圆 = 起点，菱形 = 第 256 步终点',fontsize=15)
fig.supxlabel('二维只保留约 29.8% 的全轨迹差异；图上接近或相交不等于高维编码接近。\nPCA 用所有起点每 16 步的 136 个快照拟合，全部 257 步均投影；坐标没有语义或地理含义。',fontsize=10)
fig.savefig(ROOT/'eight-trajectories.png',dpi=180,bbox_inches='tight')
fig.savefig(ROOT/'eight-trajectories.pdf',bbox_inches='tight')
plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(13,5.6),constrained_layout=True)
for p,color in zip(d['paths'],colors):
    axes[0].plot(range(257),p['fromStart'],color=color,label=f"起点 {p['seed']}",lw=1.8)
    axes[1].semilogy(range(256),np.maximum(p['toFinal'][:256],1e-10),color=color,lw=1.8)
axes[0].set(title='离各自起点越来越远，后期趋缓',xlabel='优化步数',ylabel='距自身起点的高维 RMSE')
axes[1].set(title='靠近各自最终落点（回看第 256 步）',xlabel='优化步数',ylabel='距自身终点的高维 RMSE（对数轴）')
for ax in axes:
    ax.grid(alpha=.15)
axes[0].legend(ncol=2,fontsize=9)
fig.suptitle('原始 65,536 维距离：不经过二维投影',fontsize=15)
fig.supxlabel('95% 路程表示累计步长的占比，不表示准确率或训练完成度；“各自趋缓”不等于终点互相合并。',fontsize=10)
fig.savefig(ROOT/'true-distance-curves.png',dpi=180,bbox_inches='tight')
plt.close(fig)
with (ROOT/'endpoint-summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps({'files':['eight-trajectories.png','eight-trajectories.pdf','true-distance-curves.png','endpoint-summary.csv'], 'summary':rows},ensure_ascii=False))
