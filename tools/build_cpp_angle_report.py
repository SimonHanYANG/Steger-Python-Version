"""Short updated report: C++ angle rule, latest timing and all real examples."""
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager
import numpy as np
from PIL import Image

from build_brief_report import draw_overlay

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'reports/neck-analysis/cpp-angle-20261105'


def main():
    font='C:/Windows/Fonts/msyh.ttc'
    fontManager.addfont(font)
    plt.rcParams['font.family']=FontProperties(fname=font).get_name()
    plt.rcParams['axes.unicode_minus']=False
    data=json.loads((OUTPUT/'analysis/results.json').read_text(encoding='utf-8'))
    runs=[json.loads((OUTPUT/name).read_text(encoding='utf-8'))
          for name in ('runtime.json','runtime-repeat.json')]
    samples=[s['processing_ms'] for r in runs for s in r['samples']]
    mean,median,p95,maximum=np.mean(samples),np.median(samples),np.percentile(samples,95),max(samples)
    steps=json.loads((OUTPUT/'step-timing.json').read_text(encoding='utf-8'))
    names=[('input_mask','输入整理','ROI转灰度，整理颈部mask'),
           ('gaussian','高斯导数','计算一阶、二阶灰度导数'),
           ('hessian_taylor','方向与初定位','Hessian法向与Taylor初始位置'),
           ('subpixel','亚像素精修','插值与Newton迭代定位'),
           ('linking','中心线连接','按方向和阈值连接线点'),
           ('candidate_fits','候选角度统计','10点法向角均值及极差'),
           ('final_angle','选线与质量判断','选择中心线，返回角度和标记'),
           ('handoff','数据与进程交接','共享内存、通信和等待')]
    imgs=OUTPUT/'images';imgs.mkdir(exist_ok=True)
    overview,table,cards,csv_rows=[],[],[],[]
    states={'estimated':'可测估计','review_required':'需复核','insufficient_segment':'未给出完整角度'}
    for row in data['results']:
        identifier=row['id']
        saved=json.loads((OUTPUT/'analysis'/identifier/'result.json').read_text(encoding='utf-8'))
        with Image.open(row['image_path']) as im:image=np.array(im)
        with Image.open(row['mask_path']) as im:mask=np.array(im)!=0
        raw=saved['detector']['contours'][row['contour_index']]
        points=np.column_stack([raw['row'],raw['col']])
        m=row['measurement']
        timing=np.mean([r['per_image'][identifier]['mean_ms'] for r in runs])
        angle='—' if row['angle_deg'] is None else f"{row['angle_deg']:.2f}°"
        state=states[row['status']]
        fig,axes=plt.subplots(1,3,figsize=(9,3.4))
        axes[0].imshow(image,interpolation='nearest')
        axes[1].imshow(mask,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        draw_overlay(axes[2],image,mask,points,m)
        for ax,title in zip(axes,['原始 ROI','颈部 mask','最小/最大平滑方向'],strict=True):
            ax.set_title(title,fontsize=12)
            ax.set(xticks=[],yticks=[],xlim=(-.5,63.5),ylim=(63.5,-.5))
        fig.suptitle(f'{identifier} | {angle} | {state} | 平均 {timing:.2f} ms',fontsize=13,y=.99)
        fig.tight_layout(rect=(0,0,1,.92),w_pad=1)
        fig.savefig(imgs/f'{identifier}.png',dpi=150,facecolor='white');plt.close(fig)
        table.append(f'| [{identifier}](images/{identifier}.png) | {angle} | {state} | {timing:.2f} |')
        cards.append(f'<section><h3>{identifier} · {angle} · {state} · {timing:.2f} ms</h3><img loading="lazy" src="images/{identifier}.png" alt="{identifier}"></section>')
        csv_rows.append((identifier,row['angle_deg'],state,f'{timing:.3f}'))
        overview.append((identifier,image,mask,points,m,angle,state,timing))
    fig,axes=plt.subplots(8,4,figsize=(12,23))
    for ax,item in zip(axes.flat,overview,strict=False):
        identifier,image,mask,points,m,angle,state,timing=item
        draw_overlay(ax,image,mask,points,m)
        ax.set_title(f'{identifier} {angle} {timing:.2f} ms\n{state}',fontsize=9)
        ax.set(xticks=[],yticks=[],xlim=(-.5,63.5),ylim=(63.5,-.5))
    for ax in axes.flat[len(overview):]:ax.axis('off')
    fig.suptitle('C++ 同口径：10点法向角均值的极差',fontsize=17,y=.998)
    fig.tight_layout(rect=(0,0,1,.99),h_pad=1.3,w_pad=.6)
    fig.savefig(OUTPUT/'all-results.png',dpi=145,facecolor='white');plt.close(fig)
    counts=Counter(r['status'] for r in data['results'])
    text=(f'最新版完整计算平均 **{mean:.2f} ms/条**，中位数 {median:.2f} ms，P95 {p95:.2f} ms，最大 {maximum:.2f} ms。')
    scope=f'五个常驻进程＋本地模块；29对64×64图像，共{len(samples)}次调用。含转换、检测、测角、质量判断及通信；不含读图、分割、绘图和启动。'
    rule='法向角 → 10点float滑动平均 → 最大值减最小值 → 转为度。保留旧代码N−10窗口，不做周期角展开。'
    validation='703组独立C++对照及29张真实图像法向角对照均逐位一致；29张中心线检测输出与改动前逐位一致。'
    caution='周期边界和局部方向波动可能放大极差，这是保留旧代码行为；不代表Python与Halcon的中心线完全相同。'
    history='单帧使用measure_neck()；同一条精子的多帧中位数使用独立NeckBendingHistory，不混合不同精子。'
    stepnote=f'分步骤为单独插桩实测，合计{steps["mean_total_ms"]:.2f} ms；并行按每轮最慢尺度拆分，其余等待计入交接。'
    md=['# C++ 同口径颈部弯曲角','',rule,'',history,'',validation,'',caution,'',
        '## 流程与分步骤耗时','', '| 步骤 | 作用 | 平均 ms |','|---|---|---:|']
    md += [f'| {name} | {desc} | {steps["stage_mean_ms"][key]:.3f} |' for key,name,desc in names]
    md += ['',stepnote,'','加速：共享导数/Hessian、批量连接、Cython精修、五尺度进程并行与共享内存。','',
           '## 最新耗时','',text,'',scope,'',
           f'结果：{counts["estimated"]}张可测估计、{counts["review_required"]}张需复核、{counts["insufficient_segment"]}张无完整角度。','',
           '## 逐图结果','', '| 图像 | 弯曲角 | 状态 | 最新平均 ms |','|---|---:|---|---:|',*table,'',
           '![全部结果](all-results.png)','']
    (OUTPUT/'README.md').write_text('\n'.join(md),encoding='utf-8')
    with (OUTPUT/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f);writer.writerow(['图像','弯曲角_度','状态','最新版平均耗时_ms']);writer.writerows(csv_rows)
    hsteps=''.join(f'<tr><td>{name}</td><td>{desc}</td><td>{steps["stage_mean_ms"][key]:.3f}</td></tr>' for key,name,desc in names)
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>C++同口径颈部弯曲角</title>
<style>body{{font:16px/1.75 system-ui,"Microsoft YaHei";max-width:1080px;margin:28px auto;padding:0 18px;background:#f4f7fb;color:#183047}}section{{background:white;padding:22px;border-radius:14px;margin:20px 0}}img{{width:100%;height:auto}}td,th{{padding:9px;border-bottom:1px solid #dde5ee;text-align:left}}table{{border-collapse:collapse;width:100%}}.muted{{font-size:14px;color:#627184}}a{{color:#185fc4}}</style>
<h1>C++ 同口径颈部弯曲角</h1><section><h2>角度计算方法</h2><p>{rule}</p><p>{history}</p><p class="muted">{validation}<br>{caution}</p></section>
<section><h2>流程与每步耗时</h2><table><tr><th>步骤</th><th>作用</th><th>平均ms</th></tr>{hsteps}</table><p class="muted">{stepnote}</p><p>加速：共享中间量、批量连接、Cython精修、尺度并行和共享内存。</p></section>
<section><h2>最新耗时</h2><p><b>平均 {mean:.2f} ms/条</b>，中位数 {median:.2f} ms，P95 {p95:.2f} ms，最大 {maximum:.2f} ms。</p><p class="muted">{scope}</p><p>结果：{counts['estimated']}张可测估计、{counts['review_required']}张需复核、{counts['insufficient_segment']}张无完整角度。</p><p><a href="README.md">文字版</a> · <a href="results.csv">结果CSV</a> · <a href="all-results.png">29张总览</a></p></section>
<h2>每个精子的结果</h2><p>青色为mask边界，绿色为中心线；黄色、粉色为最小及最大平滑方向对应的10点窗口。</p>{''.join(cards)}</html>'''
    (OUTPUT/'index.html').write_text(page,encoding='utf-8')
    print(OUTPUT/'index.html')


if __name__=='__main__':main()
