"""Portable, self-contained Chinese experiment report and all 29 comparisons."""
import base64
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager
import numpy as np
from PIL import Image

from build_steger_summary import base,zoom,draw_centerline,draw_fits
from steger import Contour,measure_bending
from steger.robust_measurement import _terminal_segment

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/neck-analysis/mask-only-20261106'


def angle(value):return '未输出角度' if value is None else f'{value:.2f}°'


def reason(record):
    if record.get('angle_deg') is not None:return ''
    if not record.get('points'):return '没有连接出可用于测角的连续中心线（至少6个点）。'
    text=[]
    coverage=record.get('trusted_span_coverage',record.get('span_coverage',0.))
    if coverage<.75:text.append(f'中心线只贯穿mask范围的{coverage:.2%}，不足75%。')
    p=np.array(record['points']);n=np.array(record['normals'])
    try:
        measure_bending(Contour(p,n,np.ones(len(p))))
    except ValueError as exc:
        if '3-pixel' in str(exc):
            arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
            start,stop=_terminal_segment(p,n,arc)
            length=arc[stop-1]-arc[start]
            text.append(f'计算方向的小段只有{.2*length:.2f}像素，低于3像素要求。')
        elif 'budget' in str(exc):
            text.append('异常末端延伸过长，继续排除会超出允许范围（单端最多20%，总计最多25%）。')
        else:text.append(str(exc))
    assert text,record
    return ''.join(text)


def metrics(data):
    rows=data['results'];both=[r for r in rows if all(r[m]['angle_deg'] is not None for m in ('main','mask_only'))]
    deltas=np.array([r['mask_only']['angle_deg']-r['main']['angle_deg'] for r in both])
    return dict(main_count=sum(r['main']['angle_deg'] is not None for r in rows),
                mask_count=sum(r['mask_only']['angle_deg'] is not None for r in rows),
                lost=[r['id'] for r in rows if r['main']['angle_deg'] is not None and r['mask_only']['angle_deg'] is None],
                gained=[r['id'] for r in rows if r['main']['angle_deg'] is None and r['mask_only']['angle_deg'] is not None],
                matched_count=len(both),mean_abs_delta_deg=float(np.abs(deltas).mean()),
                max_abs_delta_deg=float(np.abs(deltas).max()),
                main_rescue_count=sum(r.get('main_rescue_used',False) for r in rows),
                mask_rescue_count=sum(r.get('mask_rescue_used',False) for r in rows),
                slower_percent=(data['statistics']['mask_only']['mean_ms']/data['statistics']['main']['mean_ms']-1)*100,
                mean_mask_pixels=float(np.mean([r['mask_pixels'] for r in rows])))


def main():
    data=json.loads((OUT/'benchmark.json').read_text(encoding='utf-8'));summary=metrics(data)
    images=OUT/'images';images.mkdir(exist_ok=True)
    font='C:/Windows/Fonts/msyh.ttc';fontManager.addfont(font)
    plt.rcParams.update({'font.family':FontProperties(fname=font).get_name(),'axes.unicode_minus':False})
    cards=[];csv_rows=[]
    for r in data['results']:
        roi=np.array(Image.open(r['image_path']));mask=np.array(Image.open(r['mask_path']))!=0
        fig,axes=plt.subplots(1,4,figsize=(15.2,4.6))
        for ax in axes:base(ax,roi,mask);zoom(ax,mask)
        for offset,mode,label in ((0,'main','main'),(2,'mask_only','仅mask内')):
            result=r[mode];p=np.asarray(result.get('points',[])).reshape(-1,2)
            ns=np.asarray(result.get('normals',[])).reshape(-1,2);measurement=result.get('measurement',{})
            if len(p):
                draw_centerline(axes[offset],p,measurement,show_points=True,normals=ns)
                draw_fits(axes[offset+1],p,measurement)
            axes[offset].set_title(f'{label}：中心线、点、法向',fontsize=10)
            axes[offset+1].set_title(f'{label}：{angle(result["angle_deg"])}',fontsize=11)
        mt=data['per_image'][r['id']]['main']['mean_ms'];nt=data['per_image'][r['id']]['mask_only']['mean_ms']
        fig.suptitle(f'{r["id"]}  |  main：{angle(r["main"]["angle_deg"])}，{mt:.2f} ms  |  仅mask内：{angle(r["mask_only"]["angle_deg"])}，{nt:.2f} ms',fontsize=13,y=.98)
        why=reason(r['mask_only']);oldwhy=reason(r['main'])
        caption=('仅mask内：'+why) if why else '青色为mask轮廓；绿色为中心线；黄点和蓝紫短线为提取点与法向；黄/粉箭头为拟合方向。'
        fig.text(.5,.025,caption,ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.08,1,.92),w_pad=.7)
        path=images/f'{r["id"]}.png';fig.savefig(path,dpi=155,facecolor='white');plt.close(fig)
        encoded=base64.b64encode(path.read_bytes()).decode('ascii')
        cards.append(f'<section id="{r["id"]}"><h3>{r["id"]} · main {angle(r["main"]["angle_deg"])} / 仅mask内 {angle(r["mask_only"]["angle_deg"])}</h3>'
                     f'<p>平均耗时：main {mt:.2f} ms / 仅mask内 {nt:.2f} ms；mask含{r["mask_pixels"]}个像素。</p>'
                     +(f'<p>仅mask内未输出原因：{html.escape(why)}</p>' if why else '')
                     +(f'<p>main未输出原因：{html.escape(oldwhy)}</p>' if oldwhy else '')
                     +f'<img loading="lazy" src="data:image/png;base64,{encoded}" alt="{r["id"]}对比图"></section>')
        csv_rows.append(dict(id=r['id'],main_angle_deg=r['main']['angle_deg'],mask_angle_deg=r['mask_only']['angle_deg'],
                             delta_deg=None if any(r[m]['angle_deg'] is None for m in ('main','mask_only')) else r['mask_only']['angle_deg']-r['main']['angle_deg'],
                             main_mean_ms=mt,mask_mean_ms=nt,mask_pixels=r['mask_pixels'],
                             main_span_coverage=r['main'].get('span_coverage'),mask_span_coverage=r['mask_only'].get('span_coverage'),
                             main_reason=oldwhy,mask_reason=why))
    with (OUT/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    s=data['statistics']
    timing_table=''.join(f'<tr><td>{label}</td><td>{s[mode]["mean_ms"]:.2f}</td><td>{s[mode]["median_ms"]:.2f}</td><td>{s[mode]["p95_ms"]:.2f}</td><td>{count}/29</td></tr>'
                         for mode,label,count in (('main','main',summary['main_count']),('mask_only','严格mask内',summary['mask_count'])))
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>严格mask内计算测试</title>
<style>body{{font:16px/1.8 system-ui,"Microsoft YaHei",sans-serif;max-width:1300px;margin:28px auto;padding:0 16px;background:#f3f6fa;color:#193248}}section{{background:white;padding:23px;border-radius:12px;margin:20px 0}}h1,h2,h3{{line-height:1.4}}h3{{font-size:18px}}img{{width:100%;height:auto}}table{{width:100%;border-collapse:collapse}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #dce4ec}}li{{margin:8px 0}}.muted{{font-size:14px;color:#657386}}</style>
<h1>严格mask内计算测试</h1><p>2026-11-06 · 分支 experiment/mask-only-20261106 · main保持不变</p>
<section><h2>结论</h2><p><strong>本次严格mask内方案平均慢{summary['slower_percent']:.1f}%，输出角度从{summary['main_count']}张减少到{summary['mask_count']}张，暂不适合替换main。</strong></p>
<p>双方都得到角度的{summary['matched_count']}张，角度变化的绝对值平均{summary['mean_abs_delta_deg']:.2f}°，最大{summary['max_abs_delta_deg']:.2f}°。这是与已确认版本的差异，没有人工真值，不能称为角度误差。</p></section>
<section><h2>“全部只算mask”如何实现</h2><ul>
<li>原图只读取mask内的灰度；mask外原图既不用于滤波，也不用于精细定位。</li>
<li>滤波只在mask内像素之间计算。缺少的外部邻居不填黑色，改由现有内部像素重新分配权重。</li>
<li>原来的整图插值准备被替换为mask内部的局部插值；每次位置修正都要求点仍在mask内。</li>
<li>计算方向、连接中心线、选线和测角沿用相同规则。5种初始平滑、必要时补试3种、亮暗两种线、最多6次修正均保留。</li>
</ul><p class="muted">mask本身仍需扫描、识别区域和建立位置索引，这属于区域管理；原图mask外的灰度不参加数值计算。该实验同时改变了滤波边界和插值定义，属于一套新实现，不是与main逐位等价的优化。</p></section>
<section><h2>完整调用耗时与结果数量</h2><table><tr><th>方案</th><th>平均ms</th><th>中位数ms</th><th>P95 ms</th><th>有最终角度</th></tr>{timing_table}</table>
<p>29对64×64图像，每个方案各测60轮，共各1740次。两种方案交替、随机顺序运行；都使用5个提前启动的进程和编译模块。</p>
<p class="muted">计时含输入整理、全部搜索、补试、测角、完整性判断和进程交接；排除读图、分割、绘图、启动和排队。包含没有最终角度的图片，避免只统计容易成功的样本。main的5.93 ms为本轮对照实测，不拿此前不同轮次的5.59 ms直接计算速度差。</p></section>
<section><h2>为什么只算更少像素反而没有更快</h2><p>mask平均只有{summary['mean_mask_pixels']:.0f}/4096个像素，但新增了内部邻居配对、边缘权重处理和局部插值求解。main有{summary['main_rescue_count']}张需要额外尝试3种平滑程度，实验版增加到{summary['mask_rescue_count']}张。本实现的这些额外工作抵消了减少像素带来的收益；不代表所有mask内实现都会更慢。</p>
<p>29张图把mask外像素改成NaN或无穷大后，实验版结果完全不变；所有候选点均在mask内；单进程与五进程完整输出一致；main的29张结果与已确认版本一致。</p></section>
<h2>全部29张结果对比</h2><p>每张图左两幅是main，右两幅是严格mask内版。mask只显示轮廓；点、法向和拟合方向分别展示。</p>{''.join(cards)}</html>'''
    (OUT/'index.html').write_text(page,encoding='utf-8')
    summary['timings']=s
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    md=['# 严格mask内计算测试','',f'分支：experiment/mask-only-20261106；main保持不变。','',
        f'本次mask内方案平均慢{summary["slower_percent"]:.1f}%，返回角度从{summary["main_count"]}张降至{summary["mask_count"]}张。','',
        '| 方案 | 平均ms | 中位数ms | P95 ms | 有角度 |','|---|---:|---:|---:|---:|',
        *[f'| {label} | {s[m]["mean_ms"]:.2f} | {s[m]["median_ms"]:.2f} | {s[m]["p95_ms"]:.2f} | {summary[key]} |' for m,label,key in [('main','main','main_count'),('mask_only','严格mask内','mask_count')]],'',
        f'双方有角度的{summary["matched_count"]}张，角度变化绝对值平均{summary["mean_abs_delta_deg"]:.2f}°，最大{summary["max_abs_delta_deg"]:.2f}°；不是人工真值误差。','',
        '原图只读取mask内灰度；滤波缺失邻居重新分配权重；插值只使用mask内字段。mask的区域识别和索引仍需扫描二值mask。','',
        '这套实验改变滤波边界与插值定义，尚未保持已确认效果。单个HTML内嵌全部图片，可单独分享。','']
    for row in csv_rows:
        md.extend([f'## {row["id"]}', '',f'main：{angle(row["main_angle_deg"])}，{row["main_mean_ms"]:.2f} ms；仅mask内：{angle(row["mask_angle_deg"])}，{row["mask_mean_ms"]:.2f} ms。','',
                   *([row['mask_reason'],''] if row['mask_reason'] else []),f'![{row["id"]}](images/{row["id"]}.png)',''])
    (OUT/'README.md').write_text('\n'.join(md),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2));print(OUT/'index.html')


if __name__=='__main__':main()
