"""Rebuild derived tables/figures; preserve every exported source record byte."""
from collections import Counter
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'src'))
from vision_memory.repro import canonical_tensor_sha256
from vision_memory.reader.open_eos import generation_diagnostics


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def write_csv(path, records):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    export = load(HERE / 'artifact_manifest.json')
    for item in export['files']:
        p = (HERE / item['path']).resolve()
        assert p.is_relative_to(HERE)
        assert p.stat().st_size == item['bytes']
        assert hashlib.sha256(p.read_bytes()).hexdigest() == item['sha256'], str(p)
        if p.suffix in ('.json', '.jsonl', '.log'):
            assert not re.search(r'gh[pousr]_[A-Za-z0-9_]{25,}|github_pat_|BEGIN [A-Z ]*PRIVATE KEY|Authorization:\s*Bearer', p.read_text(encoding='utf-8')), str(p)
    artifacts = HERE / 'artifacts'
    saved = load(artifacts / 'summary.json')
    config = load(REPO / 'configs/experiments/direct_latent_geometry.json')
    prompt_ids = list(config['target']['inputs'])
    assert saved['complete'] and saved['completed_count'] == 96
    matrix = np.load(artifacts / 'endpoints_and_initials.npz', allow_pickle=False)
    order = [str(v) for v in matrix['run_ids']]
    assert len(set(order)) == 96
    positions = {run_id: i for i, run_id in enumerate(order)}
    per_run, failures, all_endpoint_rows, metrics_by_run = [], [], [], {}
    directories = {}
    verified_scores = 0
    for spec in config['runs']:
        candidates = list(artifacts.glob('lane-*/runs/' + spec['run_id']))
        assert len(candidates) == 1
        directory = candidates[0]
        directories[spec['run_id']] = directory
        terminal, manifest = load(directory/'terminal.json'), load(directory/'manifest.json')
        assert terminal['status'] == 'completed' and terminal['optimizer_steps'] == 256
        assert manifest['training_prompts'] == ['original_open']
        i = positions[spec['run_id']]
        for key, sha in [('endpoint', terminal['endpoint_latent_sha256']), ('initial', manifest['initial_latent_sha256'])]:
            tensor = torch.from_numpy(matrix[key][i].reshape(1,4,128,128).copy())
            assert canonical_tensor_sha256(tensor) == sha, (spec['run_id'], key)
        metric = rows(directory/'metrics.jsonl')
        assert [r['optimizer_step'] for r in metric] == list(range(1,257))
        assert [r['optimizer_step'] for r in rows(directory/'latent_index.jsonl')] == list(range(257))
        assert len(rows(directory/'checkpoint_index.jsonl')) == 11
        generations = rows(directory/'generations.jsonl')
        checkpoints = rows(directory/'checkpoint_generations.jsonl')
        assert len(generations) == 15 and len(checkpoints) == 11
        assert {(r['condition'], r['prompt_id']) for r in generations} == {(c,p) for c in ['matched','blank','fixed_donor'] for p in prompt_ids}
        for row in generations + checkpoints:
            assert row['query'] == config['target']['inputs'][row['prompt_id']]
            assert row['question_trained'] == (row['prompt_id'] == 'original_open')
            assert row['gold_eos_appended'] is True
            assert generation_diagnostics(row, 'ambient', [59614]) == row['scorer']
            verified_scores += 1
        matched = {r['prompt_id']: r for r in generations if r['condition'] == 'matched'}
        progress = np.cumsum([r['update_rms'] for r in metric])
        assert progress[-1] > 0
        record = {**{k:spec[k] for k in ['run_id','distribution','seed','scale','study']},
                  'original_correct': matched['original_open']['scorer']['strict_correct'],
                  'all5_correct': all(r['scorer']['strict_correct'] for r in matched.values()),
                  'path95_step': int(np.searchsorted(progress, progress[-1]*.95))+1,
                  'final_displacement_rmse': metric[-1]['delta_from_start_rms'],
                  'elapsed_seconds': terminal['elapsed_seconds'],
                  'endpoint_latent_sha256': terminal['endpoint_latent_sha256']}
        for prompt in prompt_ids:
            row = matched[prompt]
            record[prompt+'_correct'] = row['scorer']['strict_correct']
            record[prompt+'_raw'] = row['raw']
            if not row['scorer']['strict_correct']:
                failures.append({'run_id':spec['run_id'], 'prompt_id':prompt, 'raw':row['raw'],
                                 'prefix_correct':row['scorer']['answer_prefix_token_exact'],
                                 'overgeneration':row['scorer']['overgeneration'],
                                 'eos_ce':row['eos_ce']})
        per_run.append(record)
        all_endpoint_rows.extend(generations)
        metrics_by_run[spec['run_id']] = metric
    assert len(per_run) == 96 and verified_scores == 2496
    by_id = {r['run_id']:r for r in per_run}
    assert all(by_id[r['run_id']]['all5_correct'] == r['robust_qa_pass'] for r in saved['per_run'])
    assert all(by_id[r['run_id']]['original_correct'] == r['qa_pass'] for r in saved['per_run'])
    prompt_counts = {}
    for prompt in prompt_ids:
        selected = [r for r in all_endpoint_rows if r['condition']=='matched' and r['prompt_id']==prompt]
        prompt_counts[prompt] = {'n':len(selected), 'strict_correct':sum(r['scorer']['strict_correct'] for r in selected),
                                'answer_prefix_correct':sum(r['scorer']['answer_prefix_token_exact'] for r in selected),
                                'overgeneration':sum(r['scorer']['overgeneration'] for r in selected)}
    counts = {'completed':96, 'original_correct':sum(r['original_correct'] for r in per_run),
              'all5_correct':sum(r['all5_correct'] for r in per_run),
              'unique_endpoints':len({r['endpoint_latent_sha256'] for r in per_run}),
              'failed_prompt_evaluations':len(failures), 'prompt_counts':prompt_counts,
              'path95_step':{'min':min(r['path95_step'] for r in per_run),
                             'median':float(np.median([r['path95_step'] for r in per_run])),
                             'max':max(r['path95_step'] for r in per_run)}}
    write_csv(HERE/'per_run.csv', per_run)
    if failures: write_csv(HERE/'failed_answers.csv', failures)
    (HERE/'analysis.json').write_text(json.dumps(counts, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    integrity = {'status':'passed', 'exported_files_sha_verified':len(export['files']),
                 'initial_and_endpoint_tensor_hashes_verified':192,
                 'raw_generation_scorers_recomputed':verified_scores,
                 'per_run_step_and_prompt_grids_verified':96, 'source_summary_agrees':True,
                 'scope':'Verification covers exported files, all raw scoring records, and all initial/endpoint tensors; intermediate binary tensors remain on Inspire.'}
    (HERE/'integrity_check.json').write_text(json.dumps(integrity, indent=2)+'\n', encoding='utf-8')

    # Training curves are descriptive; clipping applies only to logarithmic display.
    fig, axes = plt.subplots(1,3,figsize=(14,4), constrained_layout=True)
    for ax,key,title in zip(axes[:2], ['answer_ce_before_step','eos_ce_before_step'], ['Answer loss','EOS loss']):
        values=np.array([[x[key] for x in metrics_by_run[r]] for r in order])
        ax.fill_between(range(1,257), np.maximum(np.quantile(values,.25,axis=0),1e-9), np.maximum(np.quantile(values,.75,axis=0),1e-9), alpha=.25)
        ax.plot(range(1,257),np.maximum(np.median(values,axis=0),1e-9))
        ax.set(yscale='log', title=title+' (median / IQR)', xlabel='Optimizer update', ylabel='CE before update')
    values=np.array([np.cumsum([x['update_rms'] for x in metrics_by_run[r]]) for r in order])
    values/=values[:,-1:]
    axes[2].plot(np.arange(1,257),values.T,color='#437c9b',alpha=.12)
    axes[2].plot(np.arange(1,257),np.median(values,axis=0),color='black')
    axes[2].axhline(.95,ls='--',color='#ca6533')
    axes[2].set(title='Cumulative path length',xlabel='Optimizer update',ylabel='Fraction of final path length',ylim=(0,1.02))
    fig.savefig(HERE/'training_curves.png',dpi=150);plt.close(fig)

    end=matrix['endpoint'].astype(np.float64)
    centered=end-end.mean(0)
    gram=centered@centered.T
    eigen,vectors=np.linalg.eigh(gram);ix=np.argsort(eigen)[::-1]
    eigen=np.maximum(eigen[ix],0);vectors=vectors[:,ix]
    xy=vectors[:,:2]*np.sqrt(eigen[:2])
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    for dist in ['gaussian','uniform','sphere','rademacher','heavy_tail']:
        mask=np.array([by_id[r]['distribution']==dist for r in order])
        axes[0].scatter(xy[mask,0],xy[mask,1],s=20,label=dist,alpha=.8)
    failed=np.array([not by_id[r]['all5_correct'] for r in order])
    axes[0].scatter(xy[failed,0],xy[failed,1],s=55,facecolors='none',edgecolors='#c63d32',label='Paraphrase failure')
    axes[0].set(title='Endpoint PCA (projection only)',xlabel='PC1',ylabel='PC2');axes[0].legend(fontsize=7)
    axes[1].plot(np.arange(1,97),np.cumsum(eigen)/eigen.sum(),label='Observed endpoints')
    iso=saved['isotropic_sample_rank_reference']['cumulative_variance']
    axes[1].plot(np.arange(1,len(iso)+1),iso,label='Same N,D isotropic reference')
    axes[1].axhline(.95,ls='--',color='gray');axes[1].legend(fontsize=8)
    axes[1].set(title='Centered PCA; sample rank <= 95',xlabel='Component count',ylabel='Cumulative variance',ylim=(0,1.02))
    dot=end@end.T;norm=np.diag(dot)
    distance=np.sqrt(np.maximum(norm[:,None]+norm[None,:]-2*dot,0)/end.shape[1])
    np.fill_diagonal(distance,0)
    im=axes[2].imshow(distance,cmap='viridis');fig.colorbar(im,ax=axes[2],label='Latent RMSE')
    axes[2].set(title='Endpoint pairwise distance',xlabel='Run index (NPZ order)',ylabel='Run index (NPZ order)')
    fig.savefig(HERE/'endpoint_geometry.png',dpi=150);plt.close(fig)

    width,cell=176,202
    canvas=Image.new('RGB',(width*8,cell*12+40),'white');draw=ImageDraw.Draw(canvas)
    try: font=ImageFont.truetype(matplotlib.font_manager.findfont('DejaVu Sans Mono'),10)
    except OSError: font=ImageFont.load_default()
    draw.text((8,10),'96 endpoints | blue border: all 5 correct; red: a paraphrase failed',fill='black',font=font)
    for i,r in enumerate(per_run):
        x=(i%8)*width;y=(i//8)*cell+40
        with Image.open(directories[r['run_id']]/'step-256.png') as picture:
            thumb=ImageOps.contain(picture.convert('RGB'),(168,168))
            canvas.paste(thumb,(x+4,y+4))
        draw.rectangle((x+2,y+2,x+173,y+173),outline='#1768ac' if r['all5_correct'] else '#c63d32',width=3)
        label=r['run_id'].replace('direct-','').replace('heavy_tail','tail').replace('rademacher','radem')
        draw.text((x+3,y+179),label,fill='black',font=font)
    canvas.save(HERE/'endpoint_contact_sheet.png')

    def table_members(items):
        return f"{len(items)} | {sum(r['original_correct'] for r in items)}/{len(items)} | {sum(r['all5_correct'] for r in items)}/{len(items)}"
    prompt_table='\n'.join(f"| {p} | {v['strict_correct']}/96 | {v['answer_prefix_correct']}/96 | {v['overgeneration']} |" for p,v in prompt_counts.items())
    distribution_table='\n'.join(f"| {d} | {table_members([r for r in per_run if r['seed']<8 and r['scale']==1 and r['distribution']==d])} |" for d in ['gaussian','uniform','sphere','rademacher','heavy_tail'])
    scale_table='\n'.join(f"| {s:g} | {table_members([r for r in per_run if r['seed']<8 and r['scale']==s and r['distribution']=='gaussian'])} |" for s in [.25,.5,1,2,4])
    examples='\n'.join(f"| {r['run_id']} | {r['prompt_id']} | `{r['raw']}` |" for r in failures)
    finished=datetime.fromtimestamp(load(artifacts/'terminal.json')['finished_epoch'],timezone(timedelta(hours=8))).isoformat()
    text=f'''# 原 Direct：完整96条结果与数据记录

本报告对应原单问法 Direct 实验，训练提交 `0f704178137ee4beac952224f3175a3923d5e438`，不混入新三问法轮换实验。原实验完成时间（北京时间）：{finished}。导出时间：{export['created_utc']}。

**96/96条全部完成，原问法96/96完整答对；五种问法全部正确的终点为{counts['all5_correct']}/96。** 96个终点中有{counts['unique_endpoints']}个不同的latent。480条matched终点评测全部具有正确答案前缀，17条完整答案失败均发生在正确的ambient之后继续生成内容。

## 实验究竟验证什么

同一道ambient事实，96个不同初始final latent，各进行256次更新；训练只用original_open，四个改写仅评测。固定灰图编码中心，`z_init = reference + 0.1 × scale × RMS(reference) × epsilon`。仅latent可训练，FP32 VAE和BF16 Qwen冻结；Adam学习率0.05，损失为答案CE＋EOS CE，EOS权重1。该阶段不执行U-Net。

96条由40条五分布比较、32条额外Gaussian scale比较、24条额外Gaussian起点组成，共享单元只训练一次。所有问法保留相同的两行图片使用/短答案指令。

## 完整答题结果

| 问法 | 完整答对 | 正确答案前缀 | 多余续写 |
|---|---:|---:|---:|
{prompt_table}

完整正确使用原始greedy生成（最多32个新token）评分，不截取第一个词替代完整答案。全部失败明细见[failed_answers.csv](failed_answers.csv)。

## 配对初始化分组

分布比较固定scale=1、seed0–7，每格8条；不能把所有64条Gaussian与其他每类8条直接混为平衡比较。

| 分布 | 条数 | 原问正确 | 五问全对 |
|---|---:|---:|---:|
{distribution_table}

幅度比较固定Gaussian、seed0–7，相同seed复用噪声方向。

| scale | 条数 | 原问正确 | 五问全对 |
|---|---:|---:|---:|
{scale_table}

## 轨迹及几何

![训练曲线](training_curves.png)

每条的累计路径长度由各步update RMSE相加得到。走完最终累计路程95%的步数范围为{counts['path95_step']['min']}–{counts['path95_step']['max']}，中位数{counts['path95_step']['median']:g}；这是移动距离比例，不是训练完成度。前两幅曲线展示96条的中位数与四分位区间；对数坐标仅为显示将非正值裁到1e-9，原始数据不改。

![终点几何](endpoint_geometry.png)

原汇总的终点中心化PCA达到95%方差需要{saved['all_endpoint']['r95']}个分量，位移需要{saved['all_delta']['r95']}个分量。PCA只显示有限样本几何，样本秩上限95，不据此认定存在语义簇、低维流形或唯一正确编码。右图为原始终点两两RMSE，顺序与NPZ中的run_ids一致。

![96个终点图片](endpoint_contact_sheet.png)

蓝框表示五问全对，红框表示至少一个改写失败；单独PNG位于对应run目录。PNG是8位展示图，不能替代评测时的浮点RGB。完整latent保存在NPZ中，可在锁定VAE环境重建浮点图片。

## 科学结论及边界

- 本题96个已测试起点都能通过直接优化找到原问法可读的终点，说明可读终点不唯一；这不是整个latent空间可达性的证明。
- 换问法失败在本批全部表现为正确答案后的续写。事实答案前缀的迁移强于停止行为的迁移；不能笼统称为全部改写都读不出记忆。
- 每种分布/幅度的平衡单元仅8个起点，仍为同一事实，不支持跨题、跨事件泛化结论。空白/固定donor原始评测均完整保留，但相同控制图的重复生成不算新的独立样本。
- 本报告是Direct优化结果；不把它当作DreamLite/U-Net已学会生成记忆的证据。

## 上传内容与复核

| 内容 | 数量/文件 |
|---|---|
| 完成轨迹 | 96条 |
| 每步训练指标 | 24,576条metrics记录 |
| 终点生成 | 1,440条，96×5问法×3图片条件 |
| 检查点生成 | 1,056条，96×11个检查点 |
| latent/checkpoint索引 | 24,672条latent索引、1,056条checkpoint索引 |
| 初值和终点 | [endpoints_and_initials.npz](artifacts/endpoints_and_initials.npz)，96对FP32张量，含run_ids |
| 原始汇总 | [summary.json](artifacts/summary.json) |
| 逐条便览 | [per_run.csv](per_run.csv)、[analysis.json](analysis.json) |
| 完整原始记录与96张终点图 | [artifacts](artifacts/) |
| 文件SHA清单 | [artifact_manifest.json](artifact_manifest.json) |
| 本地复核结果 | [integrity_check.json](integrity_check.json) |

已逐文件复核{len(export['files'])}个导出文件SHA；NPZ内192个初值/终点张量逐一匹配原manifest/terminal的canonical SHA；全部2496条原始生成使用原评分器重新核对，逐run汇总与原summary一致。原始JSON/JSONL保持字节不变，重算结果放在单独文件中。

**未上传的大体积文件：** 每一步latent的`.pt`、带optimizer/RNG的checkpoint、`endpoint_raw.pt`内完整浮点图片、其他步骤PNG和控制图tensor。这些仍完整保存在启智，不能把GitHub记录包称为所有二进制训练文件的镜像。原始tensor文件路径与SHA见逐run的latent_index/checkpoint_index/terminal。

启智原始根目录：`{export['source_root']}`。原执行有dl-base与trust两个阶段，保留原launch/current_execution等记录；不因这次上传重写历史。读取NPZ时按run_ids关联逐run记录，endpoint/initial每行reshape为`(1,4,128,128)`。

运行`python build_report.py`可重新检查上传文件并生成图表；使用本仓库模块及NumPy/PyTorch/Matplotlib/Pillow。CPU报告构建环境用于读取和制图，不是原训练环境；原模型/依赖绑定见lane manifest。

## 17条失败原始输出

| run | 问法 | 完整原始回答 |
|---|---|---|
{examples}
'''
    (HERE/'README.md').write_text(text,encoding='utf-8')
    print(json.dumps({'integrity':integrity,'results':counts},ensure_ascii=False))


if __name__ == '__main__':
    main()
