"""Create a short Chinese algorithm/timing summary and all 29 result figures."""
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'reports/neck-analysis'
OUTPUT = BASE/'summary-20261105'


def draw_overlay(ax, image, mask, points, measurement):
    ax.imshow(image, interpolation='nearest')
    ax.contour(mask, levels=[.5], colors=['#00c6e8'], linewidths=.9)
    ax.plot(points[:, 1], points[:, 0], color='#20da69', lw=1.6)
    if measurement.get('method') == 'geometry_arclength_direction_range_v2':
        ax.lines[-1].set_color('#aaaaaa')
        lo,hi=measurement['valid_point_start'],measurement['valid_point_stop']
        ax.plot(points[lo:hi,1],points[lo:hi,0],color='#20da69',lw=1.6)
        excluded=np.asarray(measurement['excluded_indices'],dtype=int)
        if len(excluded):
            ax.scatter(points[excluded,1],points[excluded,0],marker='x',s=26,c='#e54435',zorder=8)
        s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
        for window,key,color in [('min_window_arc_px','direction_min','#ffc04a'),
                                  ('max_window_arc_px','direction_max','#f25ed7')]:
            q=np.linspace(*measurement[window],40)
            p=np.column_stack([np.interp(q,s,points[:,k]) for k in range(2)])
            center=p.mean(axis=0);direction=np.asarray(measurement[key])
            ax.plot(p[:,1],p[:,0],color=color,lw=3)
            ax.quiver(center[1],center[0],direction[1]*7,direction[0]*7,
                      angles='xy',scale_units='xy',scale=1,color=color,width=.009)
        return
    if measurement.get('method','').startswith('cpp_legacy_'):
        for window,key,color in [('min_window','min_angle_rad','#ffc04a'),
                                  ('max_window','max_angle_rad','#f25ed7')]:
            lo,hi=measurement[window]
            p=points[lo:hi]
            center=p.mean(axis=0)
            theta=measurement[key]
            ax.plot(p[:,1],p[:,0],color=color,lw=3)
            ax.quiver(center[1],center[0],7*np.cos(theta),-7*np.sin(theta),
                      angles='xy',scale_units='xy',scale=1,color=color,width=.009)
        return
    if not measurement:
        return
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    for window, key, color in [('start_window','direction_start','#ffc04a'),
                                ('end_window','direction_end','#f25ed7')]:
        q = np.linspace(*measurement[window], 40)
        p = np.column_stack([np.interp(q, s, points[:, k]) for k in range(2)])
        center = p.mean(axis=0)
        direction = np.asarray(measurement[key])
        ax.plot(p[:, 1], p[:, 0], color=color, lw=3)
        ax.quiver(center[1], center[0], direction[1]*7, direction[0]*7,
                  angles='xy', scale_units='xy', scale=1, color=color, width=.009)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    images_dir = OUTPUT/'images'
    images_dir.mkdir(exist_ok=True)
    font_path = Path('C:/Windows/Fonts/msyh.ttc')
    if font_path.exists():
        fontManager.addfont(str(font_path))
        plt.rcParams['font.family'] = FontProperties(fname=str(font_path)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    reports = [json.loads((BASE/'opt-5ms-20261105'/name).read_text(encoding='utf-8'))
               for name in ('paired.json','repeat.json')]
    full = [s['processing_ms'] for r in reports for s in r['samples'] if s['mode']=='parallel_scales']
    mean, median, p95, maximum = np.mean(full), np.median(full), np.percentile(full,95), max(full)
    step_data = json.loads((OUTPUT/'step-timing.json').read_text(encoding='utf-8'))
    steps = [
        ('input_mask','输入整理','ROI转灰度，清理mask并确定颈部范围'),
        ('gaussian','高斯导数','计算灰度的一阶、二阶导数'),
        ('hessian_taylor','方向与初定位','用Hessian确定法向，Taylor计算初始线点'),
        ('subpixel','亚像素精修','用三次插值和Newton迭代细化线点位置'),
        ('linking','中心线连接','按方向和双阈值连接线点'),
        ('candidate_fits','候选分析','检查覆盖范围，拟合候选线两端方向'),
        ('final_angle','角度与质量','选择中心线，返回弯曲角及质量标记'),
        ('handoff','数据与进程交接','共享内存复制、任务交接及等待'),
    ]
    state_names = {'estimated':'可测估计','review_required':'需复核','insufficient_segment':'未给出完整角度'}
    with (BASE/'opt-5ms-20261105/angles-and-runtime.csv').open(encoding='utf-8-sig') as f:
        results = list(csv.DictReader(f))
    cards, md_rows, overview_items = [], [], []
    for row in results:
        identifier = row['id']
        saved = json.loads((BASE/'local-20261105/analysis'/identifier/'result.json').read_text(encoding='utf-8'))
        with Image.open(saved['image_path']) as im:
            image = np.array(im)
        with Image.open(saved['mask_path']) as im:
            mask = np.array(im)!=0
        contour = saved['detector']['contours'][saved['contour_index']]
        points = np.column_stack([contour['row'],contour['col']])
        angle = '—' if not row['angle_deg'] else f'{float(row["angle_deg"]):.2f}°'
        status = state_names[row['status']]
        timing = float(row['parallel_mean_ms'])
        fig, axes = plt.subplots(1,3,figsize=(9,3.4))
        axes[0].imshow(image,interpolation='nearest')
        axes[1].imshow(mask,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
        draw_overlay(axes[2],image,mask,points,saved['measurement'])
        for ax,title in zip(axes,('原始 ROI','颈部 mask','中心线与拟合方向'),strict=True):
            ax.set_title(title,fontsize=12)
            ax.set(xticks=[],yticks=[],xlim=(-.5,image.shape[1]-.5),ylim=(image.shape[0]-.5,-.5))
        fig.suptitle(f'{identifier}  |  {angle}  |  {status}  |  平均 {timing:.2f} ms',fontsize=13,y=.99)
        fig.tight_layout(rect=(0,0,1,.92),w_pad=1)
        fig.savefig(images_dir/f'{identifier}.png',dpi=150,facecolor='white')
        plt.close(fig)
        note = '' if row['angle_deg'] else '<p class="muted">线段过短或覆盖不足，图中仅展示局部提取结果。</p>'
        cards.append(f'<section class="result" id="{identifier}"><h3>{identifier} · {angle} · {status} · {timing:.2f} ms</h3>{note}<a href="images/{identifier}.png"><img loading="lazy" src="images/{identifier}.png" alt="{identifier} 原图、mask与测量结果"></a></section>')
        md_rows.append(f'| [{identifier}](images/{identifier}.png) | {angle} | {status} | {timing:.2f} |')
        overview_items.append((identifier,image,mask,points,saved['measurement'],angle,status,timing))
    fig,axes=plt.subplots(8,4,figsize=(12,23),squeeze=False)
    for ax,item in zip(axes.flat,overview_items,strict=False):
        identifier,image,mask,points,measurement,angle,status,timing=item
        draw_overlay(ax,image,mask,points,measurement)
        ax.set_title(f'{identifier}  {angle}  {timing:.2f} ms\n{status}',fontsize=9)
        ax.set(xticks=[],yticks=[],xlim=(-.5,63.5),ylim=(63.5,-.5))
    for ax in axes.flat[len(results):]:
        ax.axis('off')
    fig.suptitle('29 个精子的颈部弯曲角结果',fontsize=18,y=.998)
    fig.tight_layout(rect=(0,0,1,.99),h_pad=1.3,w_pad=.6)
    fig.savefig(OUTPUT/'all-results.png',dpi=145,facecolor='white')
    plt.close(fig)
    step_html=''.join(f'<tr><td>{i}</td><td>{name}</td><td>{description}</td><td>{step_data["stage_mean_ms"][key]:.3f}</td></tr>' for i,(key,name,description) in enumerate(steps,1))
    speed_items = [
        '共享计算：亮、暗线复用导数和Hessian；Hessian只在mask内计算。',
        '批量连接：批量计算邻点几何，减少Python小循环。',
        '本地编译：用Cython执行插值和Newton精修，保留float64与运算顺序。',
        '尺度并行：五个常驻进程处理独立尺度，使用共享内存传图。',
    ]
    timing_text=f'最新版平均 <b>{mean:.2f} ms/条</b>，中位数 {median:.2f} ms，P95 {p95:.2f} ms，最大 {maximum:.2f} ms。'
    scope='i5-12500，29对64×64图像，3480次完整调用；五个常驻计算进程＋本地模块。包含ROI/mask到完整结果及通信，排除读图、分割、绘图和启动。'
    step_note=f'分步骤来自另一次插桩实测（{step_data["calls"]}次），含计时开销，合计 {step_data["mean_total_ms"]:.2f} ms。并行部分按每轮最慢尺度拆分，其余等待计入交接。'
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Steger 算法总结与逐图结果</title>
<style>body{{font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif;background:#f4f7fb;color:#183047;max-width:1080px;margin:28px auto;padding:0 18px}}h1,h2,h3{{line-height:1.4}}section{{background:white;padding:22px;border-radius:14px;margin:20px 0}}h3{{font-size:17px;margin:0 0 12px}}table{{width:100%;border-collapse:collapse;font-size:15px}}th,td{{padding:9px;border-bottom:1px solid #e0e7ef;text-align:left}}td:last-child{{white-space:nowrap}}.muted{{color:#627184;font-size:14px}}img{{display:block;width:100%;height:auto}}a{{color:#185fc4}}.stat{{font-size:19px}}.scroll{{overflow:auto}}li{{margin:7px 0}}</style>
<h1>Steger 算法总结与逐图结果</h1><p class="muted">perf/angle-worker-v1.0.1 基础上的本地优化版 · 2026-11-05</p>
<section><h2>算法流程与每步耗时</h2><div class="scroll"><table><thead><tr><th>步骤</th><th>处理</th><th>作用</th><th>平均 ms/条</th></tr></thead><tbody>{step_html}</tbody></table></div><p class="muted">{step_note}</p>
<p>测角方式：拟合中心线 5–25% 与 75–95% 弧长段的方向，两方向夹角为弯曲角，直线为 0°。</p></section>
<section><h2>采用的加速手段</h2><ul>{''.join('<li>'+s+'</li>' for s in speed_items)}</ul><p>保留完整尺度、双极性、迭代和质量规则。29张真实图像及合成对照的输出逐位一致，149项测试通过。</p></section>
<section><h2>最新耗时与结果概况</h2><p class="stat">{timing_text}</p><p class="muted">{scope}</p><p>26张返回主角度：10张可测估计、16张需复核；3张线段过短或覆盖不足。未加入分割并发负载，5 ms是平均目标，并非每条上限。</p><p><a href="README.md">文字版报告</a> · <a href="results.csv">结果CSV</a> · <a href="all-results.png">29张总览图</a></p></section>
<h2>每个精子的结果</h2><p>青色：mask边界；绿色：中心线；黄、粉色：两端拟合方向。角度为空时不使用局部诊断值代替。</p>{''.join(cards)}</html>'''
    (OUTPUT/'index.html').write_text(page,encoding='utf-8')
    md=['# Steger 算法总结与逐图结果','',
        '## 算法流程与每步耗时','',
        '| 步骤 | 处理 | 作用 | 平均 ms/条 |','|---|---|---|---:|']
    md += [f'| {i} | {name} | {desc} | {step_data["stage_mean_ms"][key]:.3f} |' for i,(key,name,desc) in enumerate(steps,1)]
    md += ['',step_note,'','测角：比较中心线5–25%与75–95%弧长窗口的拟合方向，直线为0°。','',
           '## 加速手段','',*['- '+s for s in speed_items],'',
           '保留完整搜索、float64、迭代和质量规则。真实及合成样本逐位一致，149项测试通过。','',
           '## 最新耗时','',f'平均 **{mean:.2f} ms/条**；中位数 {median:.2f} ms；P95 {p95:.2f} ms；最大 {maximum:.2f} ms。',
           '',scope,'','步骤为独立插桩计时；整体为无插桩基准。未模拟分割并发负载。','',
           '## 每个精子的结果','',
           '26张返回主角度（10张可测估计、16张需复核），3张不满足测量条件。点击图像名称查看原图、mask和测量图。','',
           '| 图像 | 弯曲角 | 状态 | 最新版平均 ms |','|---|---:|---|---:|',*md_rows,'',
           '![29张结果总览](all-results.png)','']
    (OUTPUT/'README.md').write_text('\n'.join(md),encoding='utf-8')
    with (OUTPUT/'results.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['图像','弯曲角_度','状态','最新版平均耗时_ms'])
        writer.writerows((r['id'],r['angle_deg'],state_names[r['status']],f'{float(r["parallel_mean_ms"]):.3f}') for r in results)
    print(OUTPUT/'index.html')


if __name__=='__main__':
    main()
