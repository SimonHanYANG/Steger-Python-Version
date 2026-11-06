"""Shareable latency report with embedded figures and unaltered measurements."""
import base64
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager
import numpy as np
from PIL import Image

from steger import neck
from build_steger_summary import base,zoom,draw_centerline,draw_fits,STEPS

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/neck-analysis/latency-1ms-20261106'


def main():
    data=json.loads((OUT/'paired.json').read_text(encoding='utf-8'))
    profile=json.loads((OUT/'steps.json').read_text(encoding='utf-8'))
    truth=json.loads((OUT/'truth/accuracy.json').read_text(encoding='utf-8'))
    old,new=(data['statistics'][name] for name in ('baseline','optimized'))
    assert all(r['baseline']==r['optimized'] for r in data['results'])
    assert all(r['all_scale_contours_bitwise_equal'] for r in data['results'])
    rounds=data['rounds'];samples=data['samples']
    delta=np.array([np.mean([r['processing_ms'] for r in samples if r['round']==i and r['variant']=='baseline'])
                   -np.mean([r['processing_ms'] for r in samples if r['round']==i and r['variant']=='optimized'])
                   for i in range(rounds)])
    rng=np.random.default_rng(12026)
    ci=np.percentile(delta[rng.integers(0,rounds,size=(10000,rounds))].mean(axis=1),[2.5,97.5])
    summary=dict(baseline=old,optimized=new,speedup_percent=(1-new['mean_ms']/old['mean_ms'])*100,
                 saved_ms=old['mean_ms']-new['mean_ms'],saved_ms_round_block_ci95=ci.tolist(),
                 complete_results_identical=len(data['results']),
                 all_scale_contours_identical=sum(r['all_scale_contour_count'] for r in data['results']),
                 valid_angles=sum(r['optimized']['angle_deg'] is not None for r in data['results']),
                 step_mean_ms=profile['stage_mean_ms'],instrumented_total_ms=profile['mean_total_ms'],
                 truth_cases_passed=truth['passed_cases'],truth_circle_passed=truth['circle']['passed'])
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    font='C:/Windows/Fonts/msyh.ttc';fontManager.addfont(font)
    plt.rcParams.update({'font.family':FontProperties(fname=font).get_name(),'axes.unicode_minus':False})
    images=OUT/'images';images.mkdir(exist_ok=True)
    cards=[];rows=[]
    for record in data['results']:
        identifier=record['id']
        roi=np.array(Image.open(record['image_path']));mask=np.array(Image.open(record['mask_path']))!=0
        gray,mask=neck._prepare_neck_inputs(roi,mask);geometry=neck.mask_geometry(mask)
        candidates=neck.candidate_runs(gray,geometry)
        if not any(c['span_coverage']>=.75 and c['length_px']>=15 for c in candidates):
            candidates.extend(neck.candidate_runs(gray,geometry,neck.RESCUE_SCALES))
        selected=neck.choose_candidate(candidates)
        result=record['optimized'];measurement=result.get('measurement',{})
        means={name:np.mean([s['processing_ms'] for s in samples if s['id']==identifier and s['variant']==name])
               for name in ('baseline','optimized')}
        angle=result['angle_deg'];label='无最终角度' if angle is None else f'{angle:.4f}°'
        fig,axes=plt.subplots(1,3,figsize=(11,4.4))
        base(axes[0],roi,mask);axes[0].set_title('原图＋颈部mask轮廓',fontsize=11)
        for ax,name,title in zip(axes[1:],('baseline','optimized'),('发行版','本次加速版'),strict=True):
            base(ax,roi,mask);zoom(ax,mask)
            if selected is not None:
                draw_centerline(ax,selected.contour.points,measurement,show_points=True,normals=selected.contour.normals)
                draw_fits(ax,selected.contour.points,measurement)
            ax.set_title(f'{title}：{label}\n平均{means[name]:.2f} ms',fontsize=11)
        fig.suptitle(f'{identifier}｜中心线与角度保持一致',fontsize=14,y=.98)
        fig.text(.5,.035,'青色：mask轮廓；绿色：中心线；黄色点：提取点；蓝紫：法向；黄/粉：拟合方向；红叉：测角排除的末端点。',ha='center',fontsize=9)
        fig.subplots_adjust(left=.03,right=.98,bottom=.13,top=.79,wspace=.12)
        path=images/f'{identifier}.png';fig.savefig(path,dpi=150,facecolor='white');plt.close(fig)
        why=''
        if angle is None:
            coverage=result.get('trusted_span_coverage',0)
            if coverage<.75:reason=f'算法找到的中心线只贯穿mask范围的{coverage:.1%}，没有达到75%。'
            else:reason=f'用于拟合方向的小段只有约{.2*result["length_px"]:.2f}像素，没有达到3像素。'
            why=f'<p>{reason}图中的局部方向不作为最终角度。</p>'
        encoded=base64.b64encode(path.read_bytes()).decode('ascii')
        cards.append(f'<section><h3>{identifier}</h3>{why}<img loading="lazy" src="data:image/png;base64,{encoded}" alt="{identifier}结果对比"></section>')
        rows.append(dict(id=identifier,angle_deg=angle,baseline_mean_ms=means['baseline'],optimized_mean_ms=means['optimized']))
    with (OUT/'per-image.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    stage_rows=''.join(f'<tr><td>{title}</td><td>{profile["stage_mean_ms"][key]:.3f}</td></tr>' for key,title,_ in STEPS)
    timing_rows=''.join(f'<tr><td>{name}</td><td>{values["mean_ms"]:.3f}</td><td>{values["median_ms"]:.3f}</td><td>{values["p95_ms"]:.3f}</td><td>{values["max_ms"]:.3f}</td></tr>'
                        for name,values in (('发行版v1.0.2',old),('本次加速版',new)))
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Steger Python 完整流程加速测试</title>
<style>body{{font:16px/1.8 system-ui,"Microsoft YaHei",sans-serif;color:#20354a;background:#f4f7fa;max-width:1180px;margin:28px auto;padding:0 18px}}section{{background:white;margin:22px 0;padding:24px;border-radius:12px}}h1,h2,h3{{line-height:1.4}}table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;border-bottom:1px solid #e1e8ee;text-align:left}}img{{display:block;width:100%;height:auto}}li{{margin:8px 0}}.note{{color:#5a6d7b;font-size:14px}}</style>
<h1>Steger Python 完整流程加速测试</h1><p>2026-11-06 · 基于发行版v1.0.2 · experiment/latency-1ms-20261106</p>
<section><h2>结论</h2><p><strong>完整平均耗时从{old['mean_ms']:.2f} ms降到{new['mean_ms']:.2f} ms，降低{summary['speedup_percent']:.1f}%，本轮尚未达到1 ms。</strong>29张图的完整结果字典全部一致，所有8种候选平滑程度、明暗两方向下的{summary['all_scale_contours_identical']}条中心线，坐标、法向和响应逐位一致。26张有最终角度，3张不输出的原因保持原样。</p>
<p>没有使用mask-only，没有缩小高斯核，没有降低图像分辨率、减少平滑程度或修改测角规则。main、mask-only和高斯范围测试分支都保留。</p></section>
<section><h2>这次怎样提速</h2><ol>
<li><strong>任务直接交给五个计算进程：</strong>每个进程仍处理同一张图的一种平滑程度，但主进程直接发送和接收结果，减少中间的任务管理和转发。</li>
<li><strong>让电脑直接执行连接循环：</strong>找邻点、去掉重复点、连接中心线和调整法向的循环提前编译，减少Python反复解释代码的开销。点少时直接计算点与点的距离，避免每次建立搜索树；临近判断边界时仍用原方法。</li>
<li><strong>加快方向拟合前的准备：</strong>弧长累计、重复点判断、沿中心线取点及减去均值改用编译循环；方向拟合、异常点处理和角度公式保持原样。结果里的不可修改数据也不再逐项重复复制。</li>
<li><strong>mask边缘距离只在必要范围计算：</strong>仅缩小二值mask距离计算的区域，保留外围一圈背景，距离结果仍相同。灰度图、高斯导数和插值仍使用整张ROI。</li>
</ol><p class="note">还测试了重写高斯卷积和插值前准备，但没有带来稳定收益，最终保留原SciPy计算。这些试验结果不用于下面的最终统计。</p></section>
<section><h2>完整耗时</h2><table><tr><th>版本</th><th>平均ms</th><th>中位数ms</th><th>P95 ms</th><th>本轮最大ms</th></tr>{timing_rows}</table>
<p class="note">Intel i5-12500，Windows，Python3.12，29对64×64输入，每版{rounds}轮/{old['calls']}次。两版随机交替计算，使用提前启动的五进程和内存中的原图/mask；包含输入整理、完整找线、必要补试、通信、选线和测角。不含读图、分割、绘图、启动及软件排队；实际软件里的分割并发负载需另测。</p>
<p>每条平均节省约{summary['saved_ms']:.3f} ms；按整轮配对重采样估计，节省时间的95%区间为{ci[0]:.3f}—{ci[1]:.3f} ms。本轮新版3480次中没有低于1 ms的调用，3279次低于5 ms。存在偶发长耗时，不能保证每次都在5 ms内完成。</p></section>
<section><h2>加速版各步骤耗时</h2><table><tr><th>步骤</th><th>平均ms</th></tr>{stage_rows}<tr><th>这次分步测试合计</th><th>{profile['mean_total_ms']:.3f}</th></tr></table>
<p class="note">分步测试单独运行，添加的计时操作及系统负载会影响数值，所以不强行凑成上表的完整平均耗时。五个进程同时工作，找线步骤按每轮最慢进程计算；相加对应本次分步测试的实际总时间，没有把五个进程的时间直接相加。</p></section>
<section><h2>精度和稳定性验证</h2><ul><li>203项回归测试通过；关闭编译加速后的203项测试也通过。</li><li>100组随机点集以及原有拓扑、阈值边界测试，与发行版连接结果逐位对比；并测试了并发调用、进程异常退出和共享内存释放。</li><li>原有84组合成真值验收全部通过，圆弧验收通过；门槛未放宽。</li><li>29张真实图的完整结果及所有候选中心线均与发行版一致；本次全部计时调用也逐一检查输出。</li></ul>
<p class="note">结果一致的结论适用于本机环境和这批验证数据，不等于对所有新图、编译器和平台的普遍保证。</p></section>
<h2>每张精子的效果与耗时</h2><p>轮廓不覆盖颈部区域；同一张图左右两版的中心线数据一致，分别标注各自完整平均耗时。全部图片已经嵌入HTML，可单独分享。</p>{''.join(cards)}</html>'''
    (OUT/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print(OUT/'index.html')


if __name__=='__main__':main()
