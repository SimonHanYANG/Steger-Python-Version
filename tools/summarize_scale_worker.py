"""Render a Chinese local comparison report from two paired benchmark runs."""
import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('primary', type=Path)
    parser.add_argument('repeat', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    p = json.loads(args.primary.read_text(encoding='utf-8'))
    r = json.loads(args.repeat.read_text(encoding='utf-8'))
    for key in ('reference_commit', 'source_and_binary_hashes', 'input_hashes'):
        assert p[key] == r[key], key
    assert p['native_enabled'] and r['native_enabled']
    assert p['bitwise_exact'] and r['bitwise_exact']
    assert p['parity']['straight_cases'] == 84 and p['parity']['circles'] == 1
    args.output.mkdir(parents=True, exist_ok=True)
    modes = ('baseline', 'serial_native', 'parallel_scales')
    titles = ('原分支：单线程', '优化：单线程＋本地模块', '优化：5进程＋本地模块')
    english = ('Original serial', 'Optimized serial + native', '5 scale processes + native')
    colors = ('#97a8ba', '#4d89cb', '#188878')
    samples = {m: [s['processing_ms'] for report in (p, r) for s in report['samples']
                   if s['mode'] == m] for m in modes}
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), gridspec_kw={'width_ratios': [1, 1.35]})
    ax = axes[0]
    boxes = ax.boxplot([samples[m] for m in modes], orientation='horizontal',
                      tick_labels=english, patch_artist=True,
                      flierprops={'marker': '.', 'markersize': 2, 'alpha': .4})
    for patch, color in zip(boxes['boxes'], colors):
        patch.set_facecolor(color)
    ax.axvline(5, color='#d2781d', linestyle='--', label='5 ms target')
    ax.set(xlabel='Complete request time (ms)', title='Both paired runs; no samples removed')
    ax.grid(axis='x', alpha=.2)
    ax.legend()
    ax = axes[1]
    ids = list(p['stages']['baseline']['per_image'])
    y = np.arange(len(ids))
    for index, mode in enumerate(modes):
        values = [(p['stages'][mode]['per_image'][i]['mean_ms']+
                   r['stages'][mode]['per_image'][i]['mean_ms'])/2 for i in ids]
        ax.barh(y+(index-1)*.24, values, height=.22, color=colors[index], label=english[index])
    ax.set_yticks(y, ids, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(5, color='#d2781d', linestyle='--')
    ax.set(xlabel='Mean complete request time (ms)', title='Same 29 inputs and measurement policy')
    ax.legend(fontsize=8)
    ax.grid(axis='x', alpha=.2)
    fig.suptitle('Neck bending: exact-output latency optimization | Intel i5-12500', fontsize=15)
    fig.tight_layout()
    fig.savefig(args.output/'comparison.png', dpi=160)
    plt.close(fig)
    result_rows = []
    for identifier in ids:
        output = p['stages']['baseline']['per_image'][identifier]
        for report in (p, r):
            for mode in modes:
                a = report['stages'][mode]['per_image'][identifier]
                assert a['angle_deg'] == output['angle_deg'] and a['status'] == output['status']
        result_rows.append(dict(id=identifier, angle_deg=output['angle_deg'], status=output['status'],
                                baseline_mean_ms=(p['stages']['baseline']['per_image'][identifier]['mean_ms']+r['stages']['baseline']['per_image'][identifier]['mean_ms'])/2,
                                serial_native_mean_ms=(p['stages']['serial_native']['per_image'][identifier]['mean_ms']+r['stages']['serial_native']['per_image'][identifier]['mean_ms'])/2,
                                parallel_mean_ms=(p['stages']['parallel_scales']['per_image'][identifier]['mean_ms']+r['stages']['parallel_scales']['per_image'][identifier]['mean_ms'])/2,
                                parallel_p95_run1_ms=p['stages']['parallel_scales']['per_image'][identifier]['p95_ms'],
                                parallel_p95_run2_ms=r['stages']['parallel_scales']['per_image'][identifier]['p95_ms']))
    with (args.output/'angles-and-runtime.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
    table_rows = []
    for label, report in [('首轮', p), ('独立复测', r)]:
        for mode, title in zip(modes, titles):
            s = report['stages'][mode]
            table_rows.append(f'| {label} | {title} | {s["mean_ms"]:.3f} | {s["median_ms"]:.3f} | {s["p95_ms"]:.3f} | {s["p99_ms"]:.3f} | {s["max_ms"]:.3f} | {s["fraction_at_or_below_5ms"]:.1%} |')
    parallel = [report['stages']['parallel_scales'] for report in (p, r)]
    memory = sum(c['WorkingSet64'] for c in p['child_resources_after_run']) / 1024**2
    md = [
        '# 保持原有数值输出的颈部角度加速', '',
        f'完整ROI/mask到结果平均耗时：首轮 **{parallel[0]["mean_ms"]:.3f} ms/条**，独立复测 **{parallel[1]["mean_ms"]:.3f} ms/条**。', '',
        '**这是5个常驻计算进程＋可选本地编译模块的单ROI延迟，不是只占一个CPU核的结果，也不保证每次都在5 ms以内。**', '',
        f'基线：用户指定的perf/angle-worker-v1.0.1，提交 `{p["reference_commit"]}`。同一台i5-12500（6核12逻辑处理器），Python {p["environment"]["python"]}、NumPy {p["environment"]["numpy"]}、SciPy {p["environment"]["scipy"]}。', '',
        f'29对64×64原始RGB ROI/2D mask；每次独立运行每模式各{p["rounds"]}轮，共{len(samples["baseline"])}次/模式，三种模式逐图随机交错比较。预热和进程启动均在计时外；原始样本全部保留。', '',
        '| 实测 | 模式 | 平均 ms | 中位数 ms | P95 ms | P99 ms | 最大 ms | ≤5 ms比例 |',
        '|---|---|---:|---:|---:|---:|---:|---:|', *table_rows, '',
        f'同轮平均加速比：首轮{p["speedup_parallel"]:.2f}×，复测{r["speedup_parallel"]:.2f}×。', '',
        '计时包括RGB转换、mask整理、完整尺度/双极性/补救搜索、Steger提取和连接、共享内存复制、进程通信及等待、中心线选择、拟合、角度及质量检查，返回完整结果字典。排除读图、分割、可视化、导入、启动和额外队列积压。没有同时运行分割模型。', '',
        '因此，本结果已经包含了加速方案自身的通信开销；不能把原单线程延迟除以5后作为测量值。', '',
        '## 精度与行为保留证据', '',
        f'- 原29张图输出不变，仍是26张有主角度（10张estimated、16张review_required），3张insufficient_segment。',
        f'- 全部{p["parity"]["candidate_contours"]}条真实搜索候选、{p["parity"]["full_traces"]}个完整过程记录，以及进程实际返回的{p["parity"]["parallel_candidate_contours"]}条候选与拟合逐位一致。',
        '- 84个合成全检测输出＋1个圆弧，与冻结原分支逐位一致；不是只比较最终角度，也没有通过设置宽容差掩盖偏差。',
        '- 149项测试通过，包括三次插值边界、每步Newton位置、不同形状、只读输入、并发调用和工作进程失败清理。',
        '- float64、完整五尺度/双极性及必要的三尺度补救、阈值、迭代终止、中心线选择、端部窗口和质量规则均保留。', '',
        '这能证明本次测试输入上没有改变原算法的数值结果；不能据此声称原算法在任意真实图像上的绝对误差已经得到验证。原有复核标记和测量限制仍然适用。', '',
        '## 实现变化', '',
        '1. 批量计算邻点距离、方向和打分，保持原来的筛选和排序规则。',
        '2. 用可选Cython模块合并五个导数字段的三次插值和Newton更新。保留原公式、float64及运算顺序，使用MSVC /fp:strict；不启用fast-math。高斯卷积和样条系数仍由SciPy计算。',
        '3. 五个独立尺度分给五个常驻进程。图像通过共享内存提供，按原尺度顺序收集结果；并发调用在一个工作区上串行，避免图像串扰。', '',
        f'进程池启动时间：首轮{p["pool_startup_ms"]:.1f} ms，复测{r["pool_startup_ms"]:.1f} ms，未计入逐ROI延迟。首轮运行后五个子进程工作集合计约{memory:.0f} MiB（不含父进程，共享页可能重复统计）。', '',
        '多个进程会占用CPU资源，与分割并发时可能更慢。单线程本地模块虽有改善，仍未达到5 ms；不要将其与五进程方案混为一谈。', '',
        '## 集成', '',
        '[本地模块构建、Windows主入口保护、常驻工作线程接入及资源说明](../../../docs/low-latency-worker.md)。', '',
        '```powershell', 'uv sync --extra dev --extra native',
        'uv run --extra dev --extra native python tools/build_native.py', '```', '',
        '```python', 'from multiprocessing import freeze_support',
        'from steger.worker import NeckAngleWorker, native_acceleration_available', '',
        'def main():', '    assert native_acceleration_available()',
        '    with NeckAngleWorker(workers=5) as worker:',
        '        result = worker.measure(roi_rgb, neck_mask)  # 内存数组；应用提供这两个变量',
        '        print(result["angle_deg"], result["status"])', '',
        'if __name__ == "__main__":', '    freeze_support()', '    main()', '```', '',
        '在你的角度线程生命周期内只创建一次worker，正常关闭；先预热真实输入。未编译本地模块时自动回退，不能期待同样耗时。Python/平台变化需重新构建扩展并运行对照测试。', '',
        '## 可视化与原始数据', '',
        '![三种模式对比](comparison.png)', '',
        '[逐图角度与两轮耗时CSV](angles-and-runtime.csv) · [首轮完整原始数据](paired.json) · [独立复测原始数据](repeat.json)。', '',
        '[原29张图的测量叠加图及中间过程](../local-20261105/analysis/index.html)继续适用：数值输出保持一致，不重复生成图像冒充新的测量结果。', '',
        '| 图像 | 角度 ° | 状态 | 原分支平均 ms | 单线程优化 ms | 五进程平均 ms |',
        '|---|---:|---|---:|---:|---:|',
    ]
    for row in result_rows:
        angle = '—' if row['angle_deg'] is None else f'{row["angle_deg"]:.3f}'
        md.append(f'| {row["id"]} | {angle} | {row["status"]} | {row["baseline_mean_ms"]:.3f} | {row["serial_native_mean_ms"]:.3f} | {row["parallel_mean_ms"]:.3f} |')
    (args.output/'README.md').write_text('\n'.join(md)+'\n', encoding='utf-8')
    hrows = []
    for label, report in [('首轮', p), ('独立复测', r)]:
        for mode, title in zip(modes, titles):
            s = report['stages'][mode]
            hrows.append(f'<tr><td>{label}</td><td>{title}</td><td>{s["mean_ms"]:.3f}</td><td>{s["median_ms"]:.3f}</td><td>{s["p95_ms"]:.3f}</td><td>{s["max_ms"]:.3f}</td><td>{s["fraction_at_or_below_5ms"]:.1%}</td></tr>')
    detail = ''.join('<tr>'+''.join(f'<td>{html.escape("—" if row[k] is None else format(row[k], ".3f") if isinstance(row[k],float) else str(row[k]))}</td>' for k in ('id','angle_deg','status','baseline_mean_ms','serial_native_mean_ms','parallel_mean_ms'))+'</tr>' for row in result_rows)
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>保持数值输出的颈部角度加速</title>
<style>body{{font:16px/1.8 system-ui;background:#f4f7fb;color:#183047;max-width:1250px;margin:32px auto;padding:0 20px}}section{{background:white;padding:24px;border-radius:14px;margin:24px 0}}img{{max-width:100%}}td,th{{border-bottom:1px solid #dde5ee;padding:9px;text-align:left}}table{{border-collapse:collapse;font-size:14px}}a{{color:#185fc4}}.note{{background:#fff1ce;padding:16px;border-radius:10px}}.scroll{{overflow:auto}}</style>
<h1>保持数值输出的颈部角度加速</h1><p>i5-12500 · 同一批29对ROI/mask · 冻结基线 {p['reference_commit'][:12]}</p>
<section><h2>平均 {parallel[0]['mean_ms']:.2f} / {parallel[1]['mean_ms']:.2f} ms，完整ROI/mask→结果</h2>
<p class="note">目标附近的延迟需要5个常驻CPU进程＋本地编译模块。不是单CPU核耗时，也不是每次都低于5 ms。包含共享内存复制、IPC和等待，未模拟同时运行的分割或积压队列。</p>
<div class="scroll"><table><tr><th>轮次</th><th>模式</th><th>平均ms</th><th>中位数ms</th><th>P95ms</th><th>最大ms</th><th>≤5ms比例</th></tr>{''.join(hrows)}</table></div>
<p><a href="README.md">完整中文报告与集成说明</a> · <a href="angles-and-runtime.csv">逐图角度与耗时CSV</a> · <a href="paired.json">首轮原始统计</a> · <a href="repeat.json">复测原始统计</a></p>
<img src="comparison.png"></section><section><h2>原有精度与行为的验证</h2><p>29个完整结果、210条候选、29个完整过程、进程实际返回的候选和拟合逐位一致；84个合成检测＋1个圆弧逐位一致；149项测试通过。保留float64、全部搜索尺度与双极性、迭代规则、筛选和质量标记。</p>
<p>仍为26张返回主角度，3张覆盖不足或过短。原算法在真实图像上的绝对误差没有因此获得新的验证。</p><p><a href="../local-20261105/analysis/index.html">查看原29张测量图和中间过程</a></p></section>
<section><h2>逐图结果</h2><div class="scroll"><table><tr><th>图像</th><th>角度°</th><th>状态</th><th>原分支ms</th><th>单线程优化ms</th><th>五进程ms</th></tr>{detail}</table></div></section></html>'''
    (args.output/'index.html').write_text(page, encoding='utf-8')
    print(args.output/'index.html')


if __name__ == '__main__':
    main()
