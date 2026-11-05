"""Benchmark the unmodified branch API with resident ROI/mask, then render a report.

Run only after tools/analyze_necks.py has produced a local analysis directory.
No saved scale, polarity or contour is supplied to the production API.
"""

import argparse
import cProfile
import csv
import hashlib
import html
import json
import os
import platform
import pstats
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import scipy
from PIL import Image

from steger import measure_neck
from steger.io import write_json
from steger.measurement import measure_bending
from steger.types import Contour

ROOT = Path(__file__).resolve().parents[1]


def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(calls=len(a), mean_ms=float(a.mean()), median_ms=float(np.median(a)),
                p95_ms=float(np.percentile(a, 95)), p99_ms=float(np.percentile(a, 99)),
                min_ms=float(a.min()), max_ms=float(a.max()))


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def timed(function, *args):
    start = time.perf_counter_ns()
    output = function(*args)
    elapsed = (time.perf_counter_ns() - start) / 1e6
    return output, elapsed


def load_cases(analysis):
    cases = []
    for row in json.loads((analysis / 'results.json').read_text(encoding='utf-8'))['results']:
        arrays = []
        for kind in ('image', 'mask'):
            path = Path(row[kind + '_path'])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row[kind + '_sha256']
            with Image.open(path) as im:
                a = np.array(im)
            a.flags.writeable = False
            arrays.append(a)
        contour = None
        if row.get('angle_deg') is not None:
            saved = json.loads((analysis / row['id'] / 'result.json').read_text(encoding='utf-8'))
            raw = saved['detector']['contours'][row['contour_index']]
            contour = Contour(
                np.column_stack((raw['row'], raw['col'])),
                np.column_stack((raw['normal_row'], raw['normal_col'])),
                np.array(raw['response'], dtype=float),
                {key: np.asarray(value) for key, value in raw['attributes'].items()}, raw['closed'])
        cases.append(dict(record=row, roi=arrays[0], mask=arrays[1], contour=contour))
    return cases


def verify(output, row):
    assert output['status'] == row['status'], row['id']
    if row['angle_deg'] is None:
        assert output['angle_deg'] is None, row['id']
    else:
        np.testing.assert_allclose(output['angle_deg'], row['angle_deg'], atol=1e-10, rtol=0)


def run(args):
    cases = load_cases(args.analysis)
    samples, angle_samples, baselines = [], [], {}
    # A persistent worker represents a software angle thread; arrivals are serial.
    # Thread startup is outside both timers; imports/decoding happened before this.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='neck-angle') as worker:
        worker.submit(lambda: None).result()
        first = cases[0]
        output, cold_ms = worker.submit(timed, measure_neck, first['roi'], first['mask']).result()
        verify(output, first['record'])
        for case in cases:
            for _ in range(2):
                output = worker.submit(measure_neck, case['roi'], case['mask']).result()
                verify(output, case['record'])
            baselines[case['record']['id']] = output
        rng = np.random.default_rng(20261105)
        for round_index in range(args.rounds):
            for index in rng.permutation(len(cases)):
                c = cases[index]
                start = time.perf_counter_ns()
                output, elapsed = worker.submit(timed, measure_neck, c['roi'], c['mask']).result()
                roundtrip = (time.perf_counter_ns() - start) / 1e6
                assert output == baselines[c['record']['id']]
                samples.append(dict(round=round_index, id=c['record']['id'],
                                    compute_ms=elapsed, submit_to_result_ms=roundtrip))
            if (round_index + 1) % 10 == 0:
                print(f'Complete ROI/mask calls: round {round_index + 1}/{args.rounds}', flush=True)
        successful = [c for c in cases if c['contour'] is not None]
        for c in successful:
            output = worker.submit(measure_bending, c['contour']).result()
            np.testing.assert_allclose(output.angle_deg, c['record']['angle_deg'], rtol=0, atol=1e-10)
        for round_index in range(40):
            for index in rng.permutation(len(successful)):
                c = successful[index]
                output, elapsed = worker.submit(timed, measure_bending, c['contour']).result()
                angle_samples.append(dict(round=round_index, id=c['record']['id'], compute_ms=elapsed))
        profiler_rows = []
        if args.profile:
            def profile():
                profiler = cProfile.Profile()
                profiler.enable()
                for _ in range(3):
                    for c in cases:
                        measure_neck(c['roi'], c['mask'])
                profiler.disable()
                profiler.dump_stats(str(args.output / 'profile.pstats'))
                return pstats.Stats(profiler)
            data = worker.submit(profile).result()
            for (filename, line, function), (primitive, calls, self_s, cumulative_s, _) in data.stats.items():
                if 'steger' in filename and filename.endswith('.py'):
                    profiler_rows.append(dict(function=f'{Path(filename).name}:{line}:{function}',
                                              calls=calls, self_ms=self_s * 1000,
                                              cumulative_ms=cumulative_s * 1000))
            profiler_rows.sort(key=lambda x: x['cumulative_ms'], reverse=True)
    per_image = []
    for c in cases:
        r = c['record']
        group = [s for s in samples if s['id'] == r['id']]
        output = baselines[r['id']]
        per_image.append(dict(id=r['id'], angle_deg=output['angle_deg'], status=output['status'],
                              local_segment_angle_deg=output.get('local_segment_angle_deg'),
                              sigma=output.get('sigma'), polarity=output.get('polarity'),
                              span_coverage=output.get('span_coverage'),
                              rescue_search=len(r['tested_scales_px']) > 5,
                              quality_flags=';'.join(output.get('quality_flags', [])),
                              **stats([s['compute_ms'] for s in group]),
                              roundtrip_mean_ms=float(np.mean([s['submit_to_result_ms'] for s in group]))))
    cpu = subprocess.check_output(
        ['powershell', '-NoProfile', '-Command',
         'Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json'],
        text=True).strip()
    source = hashlib.sha256()
    for p in sorted((ROOT / 'src/steger').glob('*.py')):
        source.update(p.name.encode())
        source.update(p.read_bytes())
    report = dict(
        measured_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        branch=git('branch', '--show-current'), commit=git('rev-parse', 'HEAD'),
        source_sha256=source.hexdigest(), benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        environment=dict(cpu=json.loads(cpu), platform=platform.platform(), python=platform.python_version(),
                         numpy=np.__version__, scipy=scipy.__version__, logical_cpus=os.cpu_count(),
                         thread_environment={k: os.environ.get(k) for k in
                                             ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}),
        input_layouts=sorted({f'ROI {c["roi"].shape} {c["roi"].dtype}; mask {c["mask"].shape} {c["mask"].dtype}' for c in cases}),
        method=dict(entrypoint='steger.neck.measure_neck(roi, neck_mask)', rounds=args.rounds,
                    warmup_calls_per_image=2, worker_threads=1,
                    includes='raw-array conversion, mask geometry, adaptive Steger scales/polarities/rescue, selection, fits, angle, quality checks, result construction',
                    excludes='disk I/O, PNG decoding, segmentation, plotting, trace capture, process/import/thread startup; queue/handoff measured separately',
                    inputs='resident original uint8 arrays, read-only; fresh adaptive search each call; no saved parameters/contours supplied',
                    workload='one persistent worker, serial arrivals, no concurrent segmentation or rendering; normal GC; untrimmed samples',
                    angle_only='diagnostic only: existing saved centerline to measure_bending; not ROI/mask integration latency'),
        image_count=len(cases), statuses=dict(Counter(r['status'] for r in per_image)),
        images_returning_angle=len(successful),
        first_compute_call=dict(id=first['record']['id'], compute_ms=cold_ms,
                                scope='one first computation in this fresh benchmark process; imports, decoding and thread startup excluded'),
        processing=stats([s['compute_ms'] for s in samples]),
        submit_to_result=stats([s['submit_to_result_ms'] for s in samples]),
        successful_processing=stats([s['compute_ms'] for s in samples if baselines[s['id']]['angle_deg'] is not None]),
        angle_only=stats([s['compute_ms'] for s in angle_samples]),
        per_image=per_image, profiler_functions=profiler_rows,
        input_hashes=[{k: c['record'][k] for k in ('id', 'image_sha256', 'mask_sha256')} for c in cases],
        validation='All 29 outputs checked against rendered batch; complete output equality checked for every repeated ROI/mask call; angle-only checked to 1e-10 deg',
    )
    for name, rows in [('raw-samples.csv', samples), ('angle-only-samples.csv', angle_samples),
                       ('per-image.csv', per_image), ('profile-functions.csv', profiler_rows)]:
        if rows:
            with (args.output / name).open('w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    write_json(args.output / 'benchmark.json', report)
    render(report, samples, args)
    print(json.dumps({k: report[k] for k in ('statuses', 'processing', 'submit_to_result', 'angle_only', 'first_compute_call')}, indent=2), flush=True)


def render(report, samples, args):
    # All plotting/imports are after the timed workload.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = report['per_image']
    colors = {'estimated': '#16836c', 'review_required': '#d3861d', 'insufficient_segment': '#bf5061'}
    fig, axes = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={'width_ratios': [1.4, 1]})
    yy = np.arange(len(rows))
    ax = axes[0]
    ax.barh(yy, [r['median_ms'] for r in rows], color=[colors.get(r['status'], '#888888') for r in rows])
    ax.scatter([r['p95_ms'] for r in rows], yy, marker='|', color='#162c47', label='P95')
    ax.set_yticks(yy, [r['id'] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set(xlabel='ROI + mask to angle / status (ms)', title='Per-image median and P95; one persistent worker')
    ax.legend()
    ax.grid(axis='x', alpha=.2)
    ax = axes[1]
    values = [r['compute_ms'] for r in samples]
    ax.hist(values, bins=45, color='#316cc4', alpha=.85)
    for name, color in [('median_ms', '#16836c'), ('p95_ms', '#d3861d'), ('p99_ms', '#bf5061')]:
        value = report['processing'][name]
        ax.axvline(value, color=color, label=f'{name[:-3]}: {value:.2f} ms')
    ax.set(xlabel='Compute time (ms)', ylabel='Calls', title=f'All {len(samples)} untrimmed calls')
    ax.legend()
    fig.suptitle('Python neck bending | perf/angle-worker-v1.0.1 | Intel i5-12500', fontsize=15)
    fig.tight_layout()
    fig.savefig(args.output / 'runtime.png', dpi=160)
    plt.close(fig)
    p, rt, a = report['processing'], report['submit_to_result'], report['angle_only']
    paragraphs = [
        '# 本机颈部弯曲角与集成耗时实测', '',
        f'分支 `{report["branch"]}`，提交 `{report["commit"]}`。实测时间 {report["measured_at"]}。', '',
        f'CPU：{report["environment"]["cpu"]["Name"]}，6 核 12 逻辑处理器；Python {report["environment"]["python"]}，NumPy {np.__version__}，SciPy {scipy.__version__}。', '',
        f'29 对 ROI/mask，每对预热 2 次，随机顺序重复 {args.rounds} 轮；完整入口累计 {p["calls"]} 次。原始输入布局：{report["input_layouts"]}。', '',
        '计时从工作线程内调用 `measure_neck(roi, mask)` 前开始，到完整结果字典返回结束。包括灰度转换、mask 清理、五个初始尺度的双极性 Steger、自适应补救、中心线选择、两端拟合与质量检查。每次从 ROI/mask 重新计算，没有复用已知尺度、极性或中心线。', '',
        '不包括文件读取、PNG 解码、ROI/mask 生成、可视化、导入和线程启动。未同时运行分割线程；线程提交至返回另测，连续积压任务造成的排队等待未模拟。输入为内存原始数组，未改动算法源码。', '',
        '| 计时口径 | 平均 ms | 中位数 ms | P95 ms | P99 ms | 最大 ms |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for title, s in [('完整 ROI/mask→角度/状态（集成参考）', p), ('提交工作线程→取回结果', rt), ('已有中心线→角度（仅诊断）', a)]:
        paragraphs.append(f'| {title} | {s["mean_ms"]:.3f} | {s["median_ms"]:.3f} | {s["p95_ms"]:.3f} | {s["p99_ms"]:.3f} | {s["max_ms"]:.3f} |')
    paragraphs += ['', f'首次计算：{report["first_compute_call"]["compute_ms"]:.3f} ms，仅一个样本，不包含导入或启动。所有计时样本均保留，未删异常值。', '',
                   f'返回主角度 {report["images_returning_angle"]}/29；状态统计：{report["statuses"]}。需复核的数值仍需检查中心线。覆盖不足的主角度为空，局部诊断值不能替代主角度。', '',
                   '弯曲角采用同一前进方向的两端正交拟合：5–25% 与 75–95% 弧长窗口，直线为 0°。它与旧 Halcon 项目的方向极差定义不同；S 形弯折可能抵消。无真实图像角度真值，结果一致性不等于精度验证。', '',
                   f'单线程理论平均吞吐约 {1000 / p["mean_ms"]:.1f} 个 ROI/s；处理 29 个 ROI 的平均服务时间合计约 {29 * p["mean_ms"]:.1f} ms。这不是全软件帧率，不能直接代表与分割同时运行时的延迟。', '',
                   '![耗时图](runtime.png)', '',
                   '完整测量叠加图与中间过程见 [图像报告](../analysis/index.html)。[原始耗时 CSV](raw-samples.csv)、[逐图结果 CSV](per-image.csv)、[环境及统计 JSON](benchmark.json)。', '',
                   '| 图像 | 主角度 ° | 局部诊断 ° | 状态 | 平均 ms | 中位数 ms | P95 ms | 最大 ms |',
                   '|---|---:|---:|---|---:|---:|---:|---:|']
    for r in rows:
        angle = '—' if r['angle_deg'] is None else f'{r["angle_deg"]:.3f}'
        local = '—' if r['local_segment_angle_deg'] is None else f'{r["local_segment_angle_deg"]:.3f}'
        paragraphs.append(f'| {r["id"]} | {angle} | {local} | {r["status"]} | {r["mean_ms"]:.3f} | {r["median_ms"]:.3f} | {r["p95_ms"]:.3f} | {r["max_ms"]:.3f} |')
    paragraphs += ['', 'Profiler 单独运行，不参与上面的耗时统计。profile-functions.csv 的 cumulative_ms 包含子调用，不能相加。', '',
                   '复现：', '```powershell',
                   f'uv run --extra dev python tools/benchmark_local_integration.py --analysis "{args.analysis}" --output "{args.output}" --rounds {args.rounds}' + (' --profile' if args.profile else ''), '```']
    (args.output / 'README.md').write_text('\n'.join(paragraphs) + '\n', encoding='utf-8')
    table = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(r[k]) if r[k] is not None else "—")}</td>' for k in ('id', 'angle_deg', 'status', 'mean_ms', 'median_ms', 'p95_ms')) + '</tr>' for r in rows)
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>角度计算集成耗时</title>
<style>body{{font:16px/1.7 system-ui;margin:32px auto;max-width:1200px;padding:0 20px;color:#183047}}img{{max-width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}table{{border-collapse:collapse}}a{{color:#185fc4}}</style>
<h1>ROI/mask → 颈部弯曲角：本机实测</h1><p>完整算法平均 <b>{p['mean_ms']:.3f} ms</b>，中位数 {p['median_ms']:.3f} ms，P95 {p['p95_ms']:.3f} ms，最大 {p['max_ms']:.3f} ms。{p['calls']} 次调用，单个常驻工作线程。</p>
<p>输入数组已在内存，包含自适应 Steger 搜索和质量检查；排除读图、分割和可视化。未模拟同时运行的分割负载。</p>
<p>主角度 {report['images_returning_angle']}/29；review_required 表示需复核，insufficient_segment 不返回完整角度。</p>
<p><a href="../analysis/index.html">逐图测量及中间过程</a> · <a href="README.md">完整报告</a> · <a href="per-image.csv">逐图角度与耗时 CSV</a> · <a href="benchmark.json">环境与统计 JSON</a></p>
<img src="runtime.png"><table><tr><th>图像</th><th>角度 °</th><th>状态</th><th>平均 ms</th><th>中位数 ms</th><th>P95 ms</th></tr>{table}</table></html>'''
    (args.output / 'index.html').write_text(page, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=60)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error('--rounds must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    run(args)
