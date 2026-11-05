"""Read-only 041/052 diagnosis: extraction parity, geometry, angles and arrows.

Counterfactual unwrap/trim results are diagnostics, never production changes or
ground-truth measurements. Original reports and algorithm files are untouched.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
import numpy as np
from scipy import ndimage

from benchmark_optimization import reference_modules
from build_brief_report import draw_overlay
from steger.io import read_image, write_json
from steger.measurement import measure_cpp_bending as measure_bending
from steger.measurement import measure_direction_angles, measure_end_direction_bending
from steger.neck import mask_geometry
from steger.types import Contour

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'reports/neck-analysis'
OUTPUT=BASE/'angle-diagnostics-20261105'
IDENTIFIERS=('sperm_041','sperm_052')


def load(identifier, directory):
    return json.loads((BASE/directory/'analysis'/identifier/'result.json').read_text(encoding='utf-8'))


def contour(record):
    c=record['detector']['contours'][record['contour_index']]
    return Contour(np.column_stack([c['row'],c['col']]),
                   np.column_stack([c['normal_row'],c['normal_col']]),np.asarray(c['response']))


def setup_image(ax, image, mask, points, zoom=True):
    ax.imshow(image,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
    ax.contour(mask,levels=[.5],colors=['#00a9c4'],linewidths=.9)
    ax.plot(points[:,1],points[:,0],color='#159459',lw=1.4)
    if zoom:
        low=np.maximum(points.min(axis=0)-4,0)
        high=np.minimum(points.max(axis=0)+4,np.array(image.shape)-1)
        # Keep the same pixels and coordinate convention in every panel.
        ax.set(xlim=(low[1],high[1]),ylim=(high[0],low[0]))
    ax.set_aspect('equal')
    ax.set(xticks=[],yticks=[])


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    font='C:/Windows/Fonts/msyh.ttc'
    fontManager.addfont(font)
    plt.rcParams['font.family']=FontProperties(fname=font).get_name()
    plt.rcParams['axes.unicode_minus']=False
    records={identifier:load(identifier,'cpp-angle-20261105') for identifier in IDENTIFIERS}
    parity={identifier:{} for identifier in IDENTIFIERS}
    for revision in ('v1.0.1','aeb2feb2cbe6e0b8601b3e0480a2a64784a63140'):
        with reference_modules(revision) as (_,batch,_,commit):
            for identifier,r in records.items():
                result=batch.detect(read_image(r['image_path']),
                                    mask_geometry(read_image(r['mask_path'],mask=True))['mask'],
                                    r['sigma'],r['polarity'],trace=True)
                old=result.contours[r['contour_index']];new=contour(r)
                equal=all(getattr(old,k).tobytes()==getattr(new,k).tobytes()
                          for k in ('points','normals','response'))
                assert equal,(revision,identifier)
                parity[identifier][revision]=dict(commit=commit,points_normals_response_bitwise_equal=equal)
    all_rows=[]
    for folder in sorted((BASE/'cpp-angle-20261105/analysis').glob('sperm_*')):
        r=json.loads((folder/'result.json').read_text(encoding='utf-8'))
        c=contour(r);angles=c.get_attribute('angle')
        all_rows.append(dict(id=r['id'],raw_range_deg=measure_bending(c).angle_deg,
                             unwrapped_range_diagnostic_deg=measure_direction_angles(np.unwrap(angles)).angle_deg,
                             period_jumps=int(np.sum(np.abs(np.diff(angles))>np.pi))))
    with (OUTPUT/'all-angle-diagnostics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    summaries={}
    for identifier,r in records.items():
        old=load(identifier,'local-20261105'); c=contour(r); oc=contour(old)
        assert c.points.tobytes()==oc.points.tobytes()
        assert c.normals.tobytes()==oc.normals.tobytes()
        original=np.load(BASE/'local-20261105/analysis'/identifier/'process/process.npz')
        trace=np.load(BASE/'cpp-angle-20261105/analysis'/identifier/'process/process.npz')
        assert original.files==trace.files
        assert all(original[k].tobytes()==trace[k].tobytes() for k in original.files)
        old_method=measure_end_direction_bending(c)
        assert old_method.angle_deg==old['angle_deg']
        new_method=measure_bending(c)
        raw=c.get_attribute('angle');degrees=np.rad2deg(raw)
        unwrapped=np.unwrap(raw)
        corrected=measure_direction_angles(unwrapped)
        count=len(raw)-10
        float64_range=np.ptp([np.mean(raw[k:k+10]) for k in range(count)])*180/np.pi
        s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(c.points,axis=0),axis=1))]
        geometry=mask_geometry(trace['mask'])
        tangent=np.gradient(c.points,axis=0)
        right=np.column_stack([tangent[:,1],-tangent[:,0]])
        right/=np.linalg.norm(right,axis=1)[:,None]
        angle_deviation=np.rad2deg(np.arccos(np.clip(np.einsum('ij,ij->i',c.normals,right),-1,1)))
        details=[]
        for i,p in enumerate(c.points):
            k=np.argmin(np.linalg.norm(trace['candidate_points']-p,axis=1))
            origin=trace['candidate_pixels'][k]
            ev=trace['eigenvalues_algebraically_sorted'][tuple(origin)]
            details.append(dict(index=i,row=float(p[0]),col=float(p[1]),
                                arclength_fraction=float(s[i]/s[-1]),normal_deg=float(degrees[i]),
                                continuous_normal_diagnostic_deg=float(np.rad2deg(unwrapped[i])),
                                normal_vs_geometric_normal_deg=float(angle_deviation[i]),
                                mask_depth_px=float(ndimage.map_coordinates(geometry['distance'],p[:,None],order=1)[0]),
                                eigenvalue_low=float(ev[0]),eigenvalue_high=float(ev[1])))
        summary=dict(id=identifier,point_count=len(c.points),sigma=r['sigma'],polarity=r['polarity'],
                     historical_extraction=parity[identifier],all_saved_trace_arrays_bitwise_equal=True,
                     before_endpoint_fit_deg=old_method.angle_deg,now_raw_cpp_range_deg=new_method.angle_deg,
                     unwrapped_range_diagnostic_deg=corrected.angle_deg,float64_raw_range_diagnostic_deg=float(float64_range),
                     trim_last_point_diagnostic_deg=measure_direction_angles(raw[:-1]).angle_deg,
                     min_window=list(new_method.min_window),max_window=list(new_method.max_window),
                     min_mean_deg=float(np.rad2deg(new_method.min_angle_rad)),
                     max_mean_deg=float(np.rad2deg(new_method.max_angle_rad)),points=details)
        summaries[identifier]=summary
        with (OUTPUT/f'{identifier}-points.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(details[0]));writer.writeheader();writer.writerows(details)
        fig,axes=plt.subplots(2,3,figsize=(15,9))
        image=read_image(r['image_path']);mask=trace['mask']
        ax=axes[0,0];setup_image(ax,image,mask,c.points)
        ax.scatter(c.cols,c.rows,s=22,facecolors='white',edgecolors='#165bb0',linewidths=.8,zorder=5)
        interesting=range(len(c.points)-5,len(c.points)) if identifier.endswith('041') else range(5)
        for i in interesting:
            ax.annotate(str(i),(c.cols[i],c.rows[i]),xytext=(3,4),textcoords='offset points',fontsize=9,color='#a21f32',zorder=6)
        ax.set_title('实际提取点（圆点）\n修改前后坐标、法向逐位一致')
        ax=axes[0,1];draw_overlay(ax,image,mask,c.points,old_method.to_dict())
        setup_image(ax,image,mask,c.points)
        ax.set_title(f'旧方法：两端坐标拟合\n{old_method.angle_deg:.2f}°')
        ax=axes[0,2];draw_overlay(ax,image,mask,c.points,new_method.to_dict())
        setup_image(ax,image,mask,c.points)
        ax.scatter(c.cols,c.rows,s=12,facecolors='white',edgecolors='#165bb0',linewidths=.6,zorder=5)
        ax.set_title(f'新方法：原始周期角均值极差\n{new_method.angle_deg:.2f}°；箭头尖不是采样点')
        ax=axes[1,0]
        ax.plot(degrees,'o-',ms=3,label='原始0–360°法向角')
        ax.plot(np.rad2deg(unwrapped),'--',lw=1.4,label='仅连续化诊断')
        ax.set(xlabel='中心线点序号（从0开始）',ylabel='方向角 / °',title='方向连续，但数值跨周期边界')
        ax.legend(fontsize=9);ax.grid(alpha=.2)
        ax=axes[1,1]
        ax.plot(np.rad2deg(new_method.smoothed_angles_rad),'o-',label='当前算术均值')
        ax.plot(np.rad2deg(corrected.smoothed_angles_rad),'o--',label='连续化后诊断')
        ax.set(xlabel='10点窗口起始序号',ylabel='平均法向角 / °',title='相同坐标，只改变角的连续性处理')
        ax.legend(fontsize=9);ax.grid(alpha=.2)
        ax=axes[1,2]
        values=[old_method.angle_deg,new_method.angle_deg,corrected.angle_deg]
        bars=ax.bar(['旧两端拟合','当前原样C++口径','连续化诊断'],values,color=['#4184ba','#c65a49','#279277'])
        ax.bar_label(bars,fmt='%.2f°',padding=4)
        ax.set(ylabel='角度 / °',title='诊断对照，非真值，也未修改算法')
        ax.set_ylim(0,max(values)*1.25);ax.grid(axis='y',alpha=.15)
        fig.suptitle(f'{identifier}：提取是否变化？误差来自哪里？',fontsize=18)
        fig.tight_layout(rect=(0,0,1,.95),h_pad=2)
        fig.savefig(OUTPUT/f'{identifier}-comparison.png',dpi=160,facecolor='white');plt.close(fig)
    write_json(OUTPUT/'diagnostics.json',dict(samples=summaries,all_images=all_rows,
        production_modified=False,note='Unwrap and endpoint removal are causal diagnostics, not validated new measurements. Native HALCON output is unavailable.'))
    md=['# 041、052 异常角度分析','',
        '**结论：没有新增中心线点。052主要是周期角平均错误；041同时有原有端部回折与周期角问题。**','',
        '修改前后、未加速v1.0.1、加速分支aeb2feb的点坐标/法向/响应均逐位一致；保存的所有导数、Hessian、Newton过程和中心线数组也逐位一致。加速和改测角没有改变这两张图的提取结果。','',
        '| 样本 | 点数 | 旧两端拟合 | 当前原始角极差 | 连续化诊断 | float64原始角极差 |',
        '|---|---:|---:|---:|---:|---:|']
    for ident,su in summaries.items():
        md.append(f'| {ident} | {su["point_count"]} | {su["before_endpoint_fit_deg"]:.2f}° | {su["now_raw_cpp_range_deg"]:.2f}° | {su["unwrapped_range_diagnostic_deg"]:.2f}° | {su["float64_raw_range_diagnostic_deg"]:.2f}° |')
    md += ['', '## 052：没有几何离群点，主要是0/360°跳变','',
        '前几个法向角是4.27°、2.37°、0.10°、357.16°、353.46°。0.10°到357.16°的真实圆周方向变化只有约2.94°；当前公式却直接按这些数值累加。第一个10点窗口均值变成242.91°，另一个窗口达到344.80°，差为101.89°。','',
        '这两个窗口的点本来平滑，最大相邻距离约1.28像素。图中向下突出的黄色部分是按错误均值画出的方向箭头，不是新提取点。仅对角序列做连续化，同一批点的10点极差变为8.97°；这只是定位原因，不是已验证的真实角度。','',
        '![052对比](sperm_052-comparison.png)','',
        '## 041：原有端部回折被新统计方式放大','',
        '共有23个点。靠近头部的20、21、22号点原本就存在。21号点位于96.52%弧长处，法向角43.71°，前一点为345.46°；两者真实轴向差58.25°，刚好通过当前连接器60°门限。该点距mask边界仅约0.55像素，法向与中心线几何法向偏差约77.61°。','',
        '21号点对应Hessian两个特征值均为正（约4.58、8.01），20号点也都是正值（约4.09、5.30）。结合位置和几何偏差，推断这里受头颈连接附近的二维灰度结构影响，线脊方向质量较差；目前连接器没有逐点法向/几何一致性或头部端点质量检查。此为诊断推断，不是人工真值判定。','',
        '旧方法的末端拟合窗口只到95%弧长，不直接使用96.52%处的采样位置，也不读取该点的Hessian法向。边界重采样仍可能通过20→21号点之间的插值受21号坐标间接影响，因此不能说旧方法完全不受端点影响。局部坐标拟合与窗口限制减弱了它的影响。新方法第12–21号点窗口直接包含43.71°，使该窗口算术均值跌到306.91°，与另一窗口342.58°相差35.68°。','',
        '注意：按旧C++的N−10边界，最后的22号点本身并未进入角度平均；不能把35.68°简单归因于最后一个坐标点。连续化诊断为7.76°；去掉最后一个点后重新套同一公式为7.43°，后者也改变了窗口数量，仅作定位证据。','',
        '![041对比](sperm_041-comparison.png)','',
        '## 为什么旧方法看起来没有问题','',
        '旧方法从坐标拟合方向，将拟合窗口限制到两端内侧5–25%和75–95%，并用一段点做最小二乘拟合；它不使用Hessian法向角的0/360°数值，因此避开周期角平均问题，也减弱了末端异常的影响。旧图没有暴露问题，不等于旧提取完全没有局部质量问题。','',
        '原始角极差和两端拟合本来就是不同指标；连续化后的7.76°/8.97°不能要求等于旧方法的15.78°/11.71°。float32改float64后差异不足0.00002°，不是本次大角度问题的原因。同类周期问题还出现在035：原始极差143.53°，连续化诊断4.65°；其余26张未发现这类超过5°的周期处理差异。','',
        '## 修正方向与验证边界','',
        '建议把“严格旧公式复现”和“适合实际测量的连续方向统计”区分清楚：先处理法向的方向一致性与周期连续性，再为041这类头部端点增加几何/灰度质量检查。不能靠盲删一个点修好052，也不应未经验证就全局裁掉固定数量的点。','',
        '本轮未修改生产算法。连续化及裁点数字仅用于因果诊断。旧C++源码没有显式unwrap，但没有原生Halcon对这两张图的法向序列，不能断言Halcon原程序也会给出同样的异常数值。此前逐位对照证明的是相同输入角序列下复现同一算术，不是对真实测角精度的验证。','',
        '- [041逐点坐标、法向与Hessian CSV](sperm_041-points.csv)',
        '- [052逐点坐标、法向与Hessian CSV](sperm_052-points.csv)',
        '- [29张图的周期跳变诊断CSV](all-angle-diagnostics.csv)',
        '- [全部数值证据JSON](diagnostics.json)','']
    (OUTPUT/'README.md').write_text('\n'.join(md),encoding='utf-8')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>041、052异常角度分析</title><style>body{font:16px/1.8 system-ui,"Microsoft YaHei";max-width:1350px;margin:30px auto;padding:0 20px;color:#183047}img{width:100%;height:auto}section{padding:22px 0;border-bottom:1px solid #ddd}table{border-collapse:collapse}td,th{padding:9px;border-bottom:1px solid #ddd}a{color:#185fc4}</style>
    <h1>041、052异常角度分析</h1><p><b>中心线没有新提取出离群点：052主要是角度周期跳变；041还存在原有端部回折。图中箭头尖不是采样点。</b></p><p>未加速v1.0.1、加速分支及修改前后的坐标、法向、响应逐位一致；所有保存的Steger中间数组也逐位一致。</p><p><a href="README.md">详细中文分析</a> · <a href="diagnostics.json">全部数值证据</a> · <a href="all-angle-diagnostics.csv">29张周期跳变诊断</a></p>'''
    page+='<table><tr><th>样本</th><th>旧两端拟合</th><th>当前原始角极差</th><th>连续化诊断（非真值）</th></tr>'
    for ident,su in summaries.items():
        page+=f'<tr><td>{ident}</td><td>{su["before_endpoint_fit_deg"]:.2f}°</td><td>{su["now_raw_cpp_range_deg"]:.2f}°</td><td>{su["unwrapped_range_diagnostic_deg"]:.2f}°</td></tr>'
    page+='</table><section><h2>052：平滑方向跨越0°/360°，算术均值产生假方向</h2><p>4.27°、2.37°、0.10°、357.16°……实际上连续；首个10点均值却变为242.91°，导致101.89°。下方黄色尖端是错误方向的箭头，不是中心线点。</p><img src="sperm_052-comparison.png"></section>'
    page+='<section><h2>041：96.5%处的异常法向进入新窗口</h2><p>21号点靠近头部，距mask边界约0.55px，法向与几何法向偏差77.61°。旧拟合仅到95%弧长，不直接统计该点法向（坐标仍可能通过插值间接影响拟合）；新窗口直接纳入该法向，并受周期跳变放大。最后22号点因旧循环边界，本身未参与平均。</p><img src="sperm_041-comparison.png"></section>'
    page+='<p>本轮仅诊断，未修改算法。连续化/裁点不是已验证修复或角度真值。没有原生Halcon输出，不能把Python的具体异常等同于旧Halcon程序的实际结果。</p></html>'
    (OUTPUT/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps({k:{q:v for q,v in s.items() if q!='points'} for k,s in summaries.items()},ensure_ascii=False,indent=2))
    print(OUTPUT/'index.html')


if __name__=='__main__':main()
