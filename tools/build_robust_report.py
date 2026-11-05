"""Build the concise Chinese all-input improvement and latency report."""
import csv
import html
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
import numpy as np
from PIL import Image

from build_brief_report import draw_overlay

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'reports/neck-analysis'
OUT=BASE/'robust-angle-20261105'
STATES={'estimated':'可测估计','review_required':'需复核','insufficient_segment':'不返回完整角度'}
FLAGS={
 'terminal_samples_excluded':'已排除不一致末端（红叉）',
 'terminal_direction_requires_review':'端部方向仍需复核',
 'window_sensitive_over_5deg':'窗口变化影响超过5°',
 'scale_sensitive_over_5deg':'尺度变化影响超过5°',
 'insufficient_comparable_scales':'可比较尺度不足',
 'ridge_near_mask_boundary':'中心线靠近mask边界',
 'large_scale_rescue_requires_review':'较大尺度补救',
 'insufficient_direction_samples':'过短或未通过几何检查',
 'insufficient_span_coverage':'原始覆盖不足75%',
 'insufficient_trusted_span_coverage':'有效覆盖不足75%',
 'nonlinear_fit_window':'局部拟合偏离直线',
 'abrupt_local_direction_change':'局部方向突变',
}


def angle(value):
    return '—' if value is None else f'{value:.2f}°'


def load_old(identifier,folder):
    r=json.loads((BASE/folder/'analysis'/identifier/'result.json').read_text(encoding='utf-8'))
    c=r['detector']['contours'][r['contour_index']]
    return r,np.column_stack([c['row'],c['col']])


def main():
    data=json.loads((OUT/'benchmark.json').read_text(encoding='utf-8'))
    steps=json.loads((OUT/'step-timing.json').read_text(encoding='utf-8'))
    fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams['font.family']=FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name()
    plt.rcParams['axes.unicode_minus']=False
    images=OUT/'images';images.mkdir(exist_ok=True)
    records=data['results'];cards=[];table=[];overview=[];csv_rows=[]
    for r in records:
        identifier=r['id']
        with Image.open(r['image_path']) as im: roi=np.array(im)
        with Image.open(r['mask_path']) as im: mask=np.array(im)!=0
        old,op=load_old(identifier,'local-20261105')
        cpp,cp=load_old(identifier,'cpp-angle-20261105')
        points=np.array(r['points']);m=r.get('measurement',{})
        ms=data['per_image'][identifier]['mean_ms']
        excluded=m.get('excluded_indices',[])
        flags='；'.join(FLAGS.get(k,k) for k in r.get('quality_flags',[])) or '未触发质量警示'
        fig,axes=plt.subplots(1,5,figsize=(14,3.55))
        axes[0].imshow(roi,interpolation='nearest')
        axes[1].imshow(mask,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        draw_overlay(axes[2],roi,mask,op,old.get('measurement',{}))
        draw_overlay(axes[3],roi,mask,cp,cpp.get('measurement',{}))
        draw_overlay(axes[4],roi,mask,points,m)
        titles=('原始ROI','原始mask',f'早期两端拟合 {angle(r["old_endpoint_angle_deg"])}',
                f'前一版原始法向 {angle(r["cpp_angle_deg"])}',f'改进后 {angle(r["angle_deg"])}')
        for ax,title in zip(axes,titles,strict=True):
            ax.set_title(title,fontsize=11)
            ax.set(xticks=[],yticks=[],xlim=(-.5,63.5),ylim=(63.5,-.5))
        fig.suptitle(f'{identifier}  |  {STATES[r["status"]]}  |  最新平均 {ms:.2f} ms',fontsize=13,y=.99)
        fig.text(.5,.03,flags,ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.08,1,.92),w_pad=.8)
        fig.savefig(images/f'{identifier}.png',dpi=150,facecolor='white');plt.close(fig)
        # A large single-result overlay for point-level visual inspection.
        fig,ax=plt.subplots(figsize=(5,5))
        draw_overlay(ax,roi,mask,points,m)
        ax.set(xticks=[],yticks=[],xlim=(-.5,63.5),ylim=(63.5,-.5))
        ax.set_title(f'{identifier}  {angle(r["angle_deg"])}\n{STATES[r["status"]]}',fontsize=12)
        fig.tight_layout();fig.savefig(images/f'{identifier}-new.png',dpi=140);plt.close(fig)
        table.append(f'<tr><td><a href="#'+identifier+'">'+identifier+'</a></td>'
                     f'<td>{angle(r["old_endpoint_angle_deg"])}</td><td>{angle(r["cpp_angle_deg"])}</td>'
                     f'<td>{angle(r["angle_deg"])}</td><td>{STATES[r["status"]]}</td><td>{ms:.2f}</td></tr>')
        cards.append(f'<section id="{identifier}"><h3>{identifier} · {angle(r["angle_deg"])} · {STATES[r["status"]]} · {ms:.2f} ms</h3>'
                     f'<p class="muted">{html.escape(flags)}。排除点索引（从0起）：{excluded or "无"}。</p>'
                     f'<a href="images/{identifier}.png"><img loading="lazy" src="images/{identifier}.png"></a>'
                     f'<a href="images/{identifier}-new.png">查看改进后大图</a></section>')
        overview.append((r,roi,mask,points,m,ms))
        csv_rows.append(dict(id=identifier,old_endpoint_angle_deg=r['old_endpoint_angle_deg'],
                             previous_cpp_angle_deg=r['cpp_angle_deg'],new_angle_deg=r['angle_deg'],
                             status=r['status'],mean_ms=ms,excluded_indices=str(excluded),
                             quality_flags=';'.join(r.get('quality_flags',[])),
                             trusted_span_coverage=r.get('trusted_span_coverage')))
    fig,axes=plt.subplots(8,4,figsize=(12,23))
    for ax,(r,roi,mask,points,m,ms) in zip(axes.flat,overview,strict=False):
        draw_overlay(ax,roi,mask,points,m)
        ax.set_title(f'{r["id"]}  {angle(r["angle_deg"])}  {ms:.2f} ms\n{STATES[r["status"]]}',fontsize=9)
        ax.set(xticks=[],yticks=[],xlim=(-.5,63.5),ylim=(63.5,-.5))
    for ax in axes.flat[len(records):]: ax.axis('off')
    fig.suptitle('全部29张：统一规则改进后的测角结果',fontsize=17,y=.999)
    fig.tight_layout(rect=(0,0,1,.99),h_pad=1.1,w_pad=.5)
    fig.savefig(OUT/'all-results.png',dpi=145,facecolor='white');plt.close(fig)
    with (OUT/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csv_rows[0]));w.writeheader();w.writerows(csv_rows)
    counts=Counter(r['status'] for r in records)
    s=data['statistics']
    timing=f'最新版平均 {s["mean_ms"]:.2f} ms/条；中位数 {s["median_ms"]:.2f} ms；P95 {s["p95_ms"]:.2f} ms；最大 {s["max_ms"]:.2f} ms。'
    scope=f'i5-12500，29对64×64 ROI/mask，随机顺序共{s["calls"]}次。5个常驻尺度进程与本地加速模块；包含完整检测、测角、质量判断及通信，排除读图、分割、绘图、进程启动和排队；未加入分割线程并发负载。'
    labels=[('input_mask','输入与mask整理'),('gaussian','高斯导数'),('hessian_taylor','Hessian与初定位'),
            ('subpixel','亚像素精修'),('linking','中心线连接'),('candidate_fits','候选几何测角'),
            ('final_angle','最终选择与质量判断'),('handoff','共享内存与进程交接')]
    step_table=''.join(f'<tr><td>{name}</td><td>{steps["stage_mean_ms"][key]:.3f}</td></tr>' for key,name in labels)
    step_note=f'步骤来自独立插桩的{steps["calls"]}次调用，合计{steps["mean_total_ms"]:.2f} ms；并行部分按每轮最慢尺度拆分，含插桩开销，不能与无插桩平均强行对齐。'
    summary=f'26张返回角度：{counts["estimated"]}张可测估计、{counts["review_required"]}张需复核；3张不足以返回完整角度（011、050、054）。'
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>颈部测角统一改进与全图对比</title>
<style>body{{font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif;color:#183047;background:#f3f6fa;max-width:1250px;margin:28px auto;padding:0 16px}}section{{background:white;padding:22px;margin:18px 0;border-radius:12px}}h1,h2,h3{{line-height:1.4}}h3{{font-size:18px}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #dde4ee;text-align:left}}.muted{{font-size:14px;color:#617187}}a{{color:#185fc4}}.scroll{{overflow:auto}}li{{margin:7px 0}}</style>
<h1>颈部测角：统一改进与全部29张对比</h1><p class="muted">perf/angle-worker-v1.0.1 基础 · 2026-11-05 · 全部样本同一规则</p>
<section><h2>改了什么</h2><ol>
<li>保留Steger原始检测，改用等弧长局部几何拟合方向；统一方向并展开周期，消除0°/360°假大角度。</li>
<li>检查异常末端，按方向不一致与回折证据排除，原始点保留并标红叉；不删除中部点，不按角度大小选线。</li>
<li>以20%弧长窗口覆盖有效中心线的5–95%，取连续方向范围，保留S形中部弯折。仍检查有效覆盖、尺度及窗口敏感性。</li>
</ol><p>端点法向轴/线段垂线差超过45°，或回折大于90°且内侧法向也不一致时触发排除；单端最多20%弧长、总保留至少75%，超限拒绝测量。15%/20%/25%窗口与不同尺度用于复核，不通过取小角度来“修正”。</p>
<p>041排除末端第21、22号点后为11.95°，仍标记需复核；052未删点，改为15.83°。035也消除了周期假角度，但得到28.51°并保留复核，036为35.87°且需复核。新法包含中部方向，因此部分角度会比早期两端拟合更大。</p></section>
<section><h2>整体检查</h2><p>{summary} 本批仅041排除两个末端点，其余样本没有端点删除；没有样本编号特例，选择的尺度、极性与轮廓编号均与前一版相同。</p>
<p>全部29张原始检测结果与历史结果逐位一致，单进程与五进程完整输出一致，关闭本地加速后完整输出也一致。合成测试覆盖直线、周期/符号翻转、旋转、反向遍历、不均匀采样、圆弧、平滑大回弯、S形和异常末端；解析TLS也与独立SVD核对。</p>
<p class="muted">缺少人工角度真值，不能据此证明29张都达到某个角度误差；035、036等需复核结果继续明确提示。新法有意修正旧C++测角缺陷，不再声称逐位复现它；原生HALCON检测等价性尚未验证。</p></section>
<section><h2>最新版耗时</h2><p><strong>{timing}</strong></p><p class="muted">{scope}</p>
<table><tr><th>算法步骤</th><th>平均 ms/条</th></tr>{step_table}</table><p class="muted">{step_note}</p>
<p>继续保留导数复用、批量邻点连接、本地亚像素迭代及五进程并行；二维TLS改为数学等价解析主轴，复用有效线段做敏感性检查。5ms是接近的平均目标，不是每条上限。</p></section>
<section><h2>全部结果</h2><p><a href="results.csv">CSV</a> · <a href="benchmark.json">结果与耗时原始数据</a> · <a href="all-results.png">29张总览</a> · <a href="README.md">文字总结</a></p><div class="scroll"><table><tr><th>图像</th><th>早期两端拟合</th><th>前一版原始法向</th><th>改进后</th><th>状态</th><th>新版平均ms</th></tr>{''.join(table)}</table></div></section>
<h2>每个精子的对比图</h2><p>青色mask，绿色有效中心线，灰色原始末端，红叉排除点；黄/粉色为最小/最大方向窗口。方向箭头不是提取点；无完整角度时，图中局部线段仅供诊断。</p>{''.join(cards)}</html>'''
    (OUT/'index.html').write_text(body,encoding='utf-8')
    md=['# 颈部测角统一改进','',
        '原始Steger提取不变；改用等弧长局部几何拟合，处理周期跳变；排除有质量证据的异常末端并保留原始点。包含中部窗口，避免S形弯折抵消。没有针对样本编号设置规则。','',
        summary,'','全部29张原始检测逐位一致，单/多进程完整输出一致。缺少人工真值，需复核标记不能视为通过。','',
        '## 最新版耗时','',timing,'',scope,'',
        '| 步骤 | 平均ms/条 |','|---|---:|',
        *[f'| {name} | {steps["stage_mean_ms"][key]:.3f} |' for key,name in labels],
        '',step_note,'','## 全部角度对比','',
        '| 图像 | 早期两端拟合 | 前一版原始法向 | 改进后 | 状态 | 新版平均ms |','|---|---:|---:|---:|---|---:|',
        *[f'| [{r["id"]}](images/{r["id"]}.png) | {angle(r["old_endpoint_angle_deg"])} | {angle(r["cpp_angle_deg"])} | {angle(r["angle_deg"])} | {STATES[r["status"]]} | {data["per_image"][r["id"]]["mean_ms"]:.2f} |' for r in records],
        '','[完整图文报告](index.html) · [逐图大图](images/) · [结果CSV](results.csv)','']
    (OUT/'README.md').write_text('\n'.join(md),encoding='utf-8')
    print(OUT/'index.html')
    print(dict(counts))


if __name__=='__main__': main()
