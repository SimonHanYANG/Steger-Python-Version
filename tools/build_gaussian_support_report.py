"""All-support speed/geometry/truth report with embedded, shareable figures."""
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
from steger.derivatives import gaussian_masks

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/neck-analysis/gaussian-support-20261106'


def number(value,places=6):return '无最终角度' if value is None else f'{value:.{places}f}°'


def summarize(data,truth):
    result={};baseline=data['statistics']['6.0']['mean_ms']
    rng=np.random.default_rng(867)
    # Paired round blocks retain the within-round/image experimental grouping.
    means={}
    for key in data['statistics']:
        means[key]=np.array([np.mean([s['processing_ms'] for s in data['samples']
                                     if s['round']==i and str(s['truncate'])==key]) for i in range(data['rounds'])])
    for t in data['supports']:
        key=str(t);rows=[r['variants'][key] for r in data['results']]
        deltas=[abs(r['comparison']['angle_delta_deg']) for r in rows if r['comparison']['angle_delta_deg'] is not None]
        geometry=[r['comparison']['selected_geometry']['max_px'] for r in rows if r['comparison']['selected_geometry']]
        paired=means['6.0']-means[key]
        boot=paired[rng.integers(0,len(paired),size=(10000,len(paired)))].mean(axis=1)
        result[key]=dict(truncate=t,example_width_sigma2=len(gaussian_masks(2.,t)[0]),
            mean_ms=data['statistics'][key]['mean_ms'],p95_ms=data['statistics'][key]['p95_ms'],
            median_ms=data['statistics'][key]['median_ms'],
            derivative_mean_ms=data['derivative_statistics_sigma2'][key]['mean_ms'],
            speedup_percent=(1-data['statistics'][key]['mean_ms']/baseline)*100,
            saved_ms_mean=float(paired.mean()),saved_ms_round_block_ci95=list(map(float,np.percentile(boot,[2.5,97.5]))),
            valid_angles=sum(r['angle_deg'] is not None for r in rows),
            angle_max_change_deg=max(deltas),angle_mean_change_deg=float(np.mean(deltas)),
            geometry_max_change_px=0. if t==6. else max(geometry),
            contour_bitwise_equal_count=sum(r['comparison']['raw_contour_bitwise_equal'] for r in rows),
            changed_point_count_cases=sum(len(r['variants'][key]['points'])!=len(r['variants']['6.0']['points']) for r in data['results']),
            changed_selection_cases=sum(r['comparison']['selection_changed'] for r in rows),
            changed_status_cases=sum(r['comparison']['status_changed'] for r in rows),
            changed_two_decimal_angle_cases=sum(a is not None and b is not None and round(a,2)!=round(b,2)
                for a,b in ((r['variants'][key]['angle_deg'],r['variants']['6.0']['angle_deg']) for r in data['results'])),
            synthetic=truth['summary'][key])
    return result


def main():
    data=json.loads((OUT/'benchmark.json').read_text(encoding='utf-8'))
    truth=json.loads((OUT/'synthetic.json').read_text(encoding='utf-8'))
    summary=summarize(data,truth)
    font='C:/Windows/Fonts/msyh.ttc';fontManager.addfont(font)
    plt.rcParams.update({'font.family':FontProperties(fname=font).get_name(),'axes.unicode_minus':False})
    images=OUT/'images';images.mkdir(exist_ok=True)
    cards=[];csv_rows=[]
    for record in data['results']:
        roi=np.array(Image.open(record['image_path']));mask=np.array(Image.open(record['mask_path']))!=0
        fig,axes=plt.subplots(2,4,figsize=(13.6,9.2))
        base(axes.flat[0],roi,mask);axes.flat[0].set_title('原始ROI＋mask轮廓',fontsize=11)
        for ax,t in zip(list(axes.flat)[1:],data['supports'],strict=True):
            key=str(t);r=record['variants'][key]
            base(ax,roi,mask);zoom(ax,mask)
            points=np.asarray(r['points']).reshape(-1,2);normals=np.asarray(r['normals']).reshape(-1,2)
            if len(points):
                draw_centerline(ax,points,r.get('measurement',{}),show_points=True,normals=normals)
                draw_fits(ax,points,r.get('measurement',{}))
            width=summary[key]['example_width_sigma2']
            actual=len(gaussian_masks(r['sigma'],t)[0]) if 'sigma' in r else None
            ms=data['per_image'][record['id']][key]['mean_ms']
            ax.set_title(f'{width}×{width}档：{number(r["angle_deg"],4)}\n本图实际{actual}×{actual}，平均{ms:.2f} ms',fontsize=10)
            csv_rows.append(dict(id=record['id'],truncate=t,example_width_sigma2=width,
                actual_selected_kernel=actual,selected_sigma=r.get('sigma'),angle_deg=r['angle_deg'],
                delta_deg=r['comparison']['angle_delta_deg'],mean_ms=ms,
                centerline_max_difference_px=None if not r['comparison']['selected_geometry'] else r['comparison']['selected_geometry']['max_px'],
                bitwise_equal=r['comparison']['raw_contour_bitwise_equal'],points=len(points),
                span_coverage=r.get('trusted_span_coverage'),status=r['status']))
        fig.suptitle(f'{record["id"]}｜仅缩小高斯导数范围，其他流程保持发行版规则',fontsize=15,y=.99)
        fig.text(.5,.02,'档位名以平滑程度2时的范围表示；每图实际范围随选中的平滑程度变化。青色mask轮廓，绿色中心线，黄点/蓝紫法向，黄/粉拟合方向。',ha='center',fontsize=9)
        fig.subplots_adjust(left=.025,right=.99,bottom=.09,top=.87,hspace=.32,wspace=.08)
        path=images/f'{record["id"]}.png';fig.savefig(path,dpi=155,facecolor='white');plt.close(fig)
        encoded=base64.b64encode(path.read_bytes()).decode('ascii')
        notices=[]
        for t in data['supports']:
            r=record['variants'][str(t)]
            if r['angle_deg'] is None:
                coverage=r.get('trusted_span_coverage',r.get('span_coverage',0.))
                if coverage<.75:notice=f'中心线只贯穿mask范围的{coverage:.2%}，不足75%'
                elif not r.get('measurement'):notice=f'计算方向的小段约{.2*r["length_px"]:.2f}像素，低于3像素要求'
                else:notice='没有足够完整的中心线'
                notices.append(notice)
        why=f'<p>未输出说明（基准档）：{html.escape(notices[0])}；图中的局部方向不作为最终角度。</p>' if notices else ''
        cards.append(f'<section id="{record["id"]}"><h3>{record["id"]}</h3>{why}'
                     f'<img loading="lazy" src="data:image/png;base64,{encoded}" alt="{record["id"]}七种范围对比"></section>')
    with (OUT/'per-image.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(csv_rows[0]));w.writeheader();w.writerows(csv_rows)
    with (OUT/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        fields=['truncate','example_width_sigma2','mean_ms','p95_ms','speedup_percent','derivative_mean_ms',
                'valid_angles','angle_max_change_deg','geometry_max_change_px','contour_bitwise_equal_count',
                'changed_point_count_cases','changed_selection_cases','changed_status_cases']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in summary.values())
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    timing_rows=''.join(f'<tr><td>{r["example_width_sigma2"]}×{r["example_width_sigma2"]}</td>'
        f'<td>{r["mean_ms"]:.3f}</td><td>{r["p95_ms"]:.3f}</td><td>{r["speedup_percent"]:.2f}%</td>'
        f'<td>{r["derivative_mean_ms"]:.3f}</td><td>{r["valid_angles"]}/29</td>'
        f'<td>{r["angle_max_change_deg"]:.8f}</td><td>{r["geometry_max_change_px"]:.8f}</td></tr>' for r in summary.values())
    truth_rows=''.join(f'<tr><td>{r["example_width_sigma2"]}×{r["example_width_sigma2"]}</td>'
        f'<td>{r["synthetic"]["corrected_profile_passed"]}/84</td><td>{"通过" if r["synthetic"]["circle"]["passed"] else "未通过"}</td>'
        f'<td>{r["synthetic"]["raw_angle_count"]}/60</td><td>{r["synthetic"]["raw_max_angle_change_deg"]:.6f}</td>'
        f'<td>{r["synthetic"]["raw_max_position_p95_change_px"]:.6f}</td></tr>' for r in summary.values())
    candidate=summary['4.0'];narrow=summary['2.5']
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>缩小高斯导数计算范围测试</title>
<style>body{{font:16px/1.8 system-ui,"Microsoft YaHei",sans-serif;color:#193248;background:#f3f6fa;max-width:1360px;margin:28px auto;padding:0 18px}}section{{background:white;margin:20px 0;padding:23px;border-radius:12px}}h1,h2,h3{{line-height:1.4}}h3{{font-size:19px}}table{{width:100%;border-collapse:collapse;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #dce4ec;text-align:left}}img{{display:block;width:100%;height:auto}}.scroll{{overflow:auto}}.muted{{font-size:14px;color:#657386}}li{{margin:8px 0}}</style>
<h1>缩小高斯导数计算范围测试</h1><p>2026-11-06 · 基于最新发行版v1.0.2 · 分支experiment/gaussian-support-20261106</p>
<section><h2>结论</h2><ul>
<li><strong>有小幅提速，但没有任何缩小档位能保持逐位相同。</strong>29张真实图的原始中心线、法向及响应，所有缩小档位均为0/29逐位一致；基准档为29/29。</li>
<li><strong>19×19是较保守的后续候选：</strong>完整平均{candidate['mean_ms']:.2f} ms，比同轮基准快{candidate['speedup_percent']:.2f}%；最大角度变化{candidate['angle_max_change_deg']:.6f}°，最大中心线位置差{candidate['geometry_max_change_px']:.8f}像素。选线、点数和状态在本批保持一致，但数值并非不变。</li>
<li>进一步缩小并不划算：13×13已出现选线变化，中心线最大位置差约{narrow['geometry_max_change_px']:.2f}像素；11×11最大角度变化超过5°。不能仅凭“仍有26张返回角度”判断精度相同。</li>
<li>默认仍保留发行版范围。若要求严格不变，继续使用27×27基准档；是否接受微小变化，需要结合你的实际角度误差要求和更多标注数据决定。</li>
</ul></section>
<section><h2>测试只改什么</h2><p>仅调整高斯核的截断范围，平滑强度、整张ROI输入、mask筛选、三次插值、精细定位、选线、测角和五进程配置都保持原规则。没有使用mask-only分支的代码。</p>
<p>27×27是平滑程度为2时的示例。各档会同步缩短其他平滑程度的范围，平滑程度本身不变。半径仍按 ceil(范围倍数×平滑程度＋0.5) 计算，最小半径2；原版的尾部权重处理不变。</p></section>
<section><h2>全部29张实图：速度和变化</h2><div class="scroll"><table><tr><th>平滑程度2时范围</th><th>完整平均ms</th><th>P95 ms</th><th>平均提速</th><th>仅导数平均ms</th><th>有角度</th><th>最大角差°</th><th>最大线位置差px</th></tr>{timing_rows}</table></div>
<p class="muted">i5-12500，29对64×64输入，每档各60轮/1740次，总计12180次。所有档位在同一个提前启动的五进程计算池内随机交替，数据已在内存；包含完整找线、补试、测角和通信，排除读图、分割、绘图、启动、排队及分割并发负载。仅导数另测每档1160次，固定平滑程度2；该项不包含完整算法。</p>
<p>导数计算本身明显减少，但它只是总耗时的一部分，因此完整流程只减少约0.08至0.26ms，平均仍未达到5ms。</p><p class="muted">位置差采用两条中心线间双向点到线段距离；它包含端点范围变化，不等于已知真值误差。角差仅比较双方有最终角度的26张；3张未输出原因仍是长度/覆盖门槛。</p></section>
<section><h2>已知真值的补充检查</h2><div class="scroll"><table><tr><th>示例范围</th><th>原84组验收</th><th>原圆弧验收</th><th>不加额外校正的直线有角度</th><th>直线最大角差°</th><th>直线位置P95最大变化px</th></tr>{truth_rows}</table></div>
<p>84组保留发行版的定位误差、线宽误差和拟合通过率门槛；圆弧仍要求半径误差P95&lt;0.01像素、角误差&lt;0.1°。这组包含原有额外校正，因此另测60条不使用线宽/截面校正的直线，与颈部找线设置一致。</p>
<p class="muted">通过原有合成验收仅表示仍在这些条件的门槛内，不代表结果逐位不变，也不证明真实精子图像角度误差没有变化。每组详细数值已保留在synthetic.json。</p></section>
<h2>每个精子的7档效果对比</h2><p>左上为原图与mask轮廓，其余按基准到最小范围排列。档位名以平滑程度2命名，每图另列实际选中核宽；全部图片已嵌入此HTML，可独立分享。</p>{''.join(cards)}</html>'''
    assert '复核' not in page
    (OUT/'index.html').write_text(page,encoding='utf-8')
    md=['# 缩小高斯导数范围测试','',
        '基于v1.0.2；不使用mask-only；main和mask-only分支保持不变。','',
        '| σ=2示例范围 | 完整平均ms | 提速 | 最大角差° | 最大中心线差px | 有角度 |','|---|---:|---:|---:|---:|---:|',
        *[f'| {r["example_width_sigma2"]}×{r["example_width_sigma2"]} | {r["mean_ms"]:.3f} | {r["speedup_percent"]:.2f}% | {r["angle_max_change_deg"]:.8f} | {r["geometry_max_change_px"]:.8f} | {r["valid_angles"]}/29 |' for r in summary.values()], '',
        '所有缩小档位都不是逐位相同；19×19是较保守的研究候选，默认仍保留6倍范围（σ=2时27×27）。没有真实角度人工真值，不能把微小变化说成精度完全不变。','',
        '计时：每档1740次完整内存ROI/mask调用，五进程，随机交替；不含读图、分割、绘图、启动、排队和分割并发负载。','',
        'HTML内嵌全部29张效果图，可单独分享。benchmark.json/synthetic.json保留完整数据；summary.csv/per-image.csv为汇总与逐图表。','']
    for record in data['results']:md += [f'## {record["id"]}','',f'![{record["id"]}](images/{record["id"]}.png)','']
    (OUT/'README.md').write_text('\n'.join(md),encoding='utf-8')
    print(json.dumps({k:{x:v for x,v in r.items() if x!='synthetic'} for k,r in summary.items()},indent=2))
    print(OUT/'index.html')


if __name__=='__main__':main()
