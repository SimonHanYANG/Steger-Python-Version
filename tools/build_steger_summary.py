"""Create the plain-language report and numerical result figures, without changing detection."""

import csv
import hashlib
import html
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.lines import Line2D
from PIL import Image

from steger import Contour, measure_bending
from steger.neck import _prepare_neck_inputs, mask_geometry

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/neck-analysis/robust-angle-20261105"
OUTPUT = ROOT / "reports/neck-analysis/steger-python-version-summary-20261105"
TITLE = "Steger Python Version 总结"
COLORS = dict(mask="#00bdd2", line="#19a85b", points="#ffe369", normal="#6660cb",
              minimum="#e89a14", maximum="#d440a0", excluded="#e1483b")

STEPS = [
    ("input_mask", "整理输入", "原图转灰度，保留最大的颈部mask区域。"),
    ("gaussian", "计算灰度变化", "平滑去噪，计算图像亮度的变化。"),
    ("hessian_taylor", "寻找中心点和法向", "找细线中心点，确定横穿细线的方向（法向）。"),
    ("subpixel", "精细定位", "把点的位置细化到小数像素。"),
    ("linking", "连成中心线", "将位置相邻、方向相近的点连起来。"),
    ("candidate_fits", "分段拟合方向", "检查线的末端，计算各小段的朝向。"),
    ("final_angle", "选线并计算角度", "选出合适的中心线，计算弯曲角。"),
    ("handoff", "数据交接", "分发图像和任务，等待并收回计算结果。"),
]

ACCELERATION = [
    ("5个进程同时找线", "同一张ROI，用5种平滑程度同时找线。每个进程负责一种平滑程度下的去噪、找点、精细定位和连接中心线，再汇总选线测角。"),
    ("减少重复计算", "寻找亮线和暗线共用计算结果；多个相邻点一起计算。"),
    ("减少Python循环开销", "把反复运行的精细定位步骤，提前转成电脑能直接执行的代码。"),
    ("复用找线进程和图像", "负责上述5种平滑程度找线的进程，提前开好并反复使用。每来一条精子，直接把ROI和mask交给它们，共用一份图像数据，省去重复启动和逐份传图的时间。"),
    ("直接用公式计算方向", "根据点的坐标直接算拟合方向，检查不同分段长度时复用已选好的中心线。"),
]

ANGLE_STEPS = [
    "从两端检查点的方向：法向短线应大致垂直于相邻点连线。若偏差超过45°，或末端回折超过90°且内侧相邻点的法向偏差也超过45°，就排除这个末端点。只删末端；每端最多删20%长度，整条至少保留75%，排除点标红叉。",
    "沿中心线依次取小段，用直线表示每段的朝向。每段长为有效中心线的20%，首尾各5%不参与拟合。",
    "比较各段朝向，取最大方向角减最小方向角，直线为0°。中间弯折也参与计算，并处理0°/360°跳变。",
]


def angle_text(value):
    return "未输出最终角度" if value is None else f"{value:.2f}°"


def numerical_diagnosis(record, mask):
    """Explain the actual failed gate with the same measured quantities as production."""
    points = np.asarray(record["points"])
    normals = np.asarray(record["normals"])
    contour = Contour(points, normals, np.ones(len(points)))
    geometry = mask_geometry(mask)
    projected = points @ geometry["axis"]
    span = float(np.ptp(projected))
    mask_span = geometry["span"]
    coverage = span / mask_span
    try:
        measured = measure_bending(contour)
        error = None
        valid_length = measured.arc_length_px
        # Saved complete record is from the exact same source snapshot.
        assert json.loads(json.dumps(measured.to_dict())) == record.get("measurement"), record["id"]
    except ValueError as exc:
        measured, error = None, str(exc)
    reasons = []
    if error:
        if error != "Neck is too short for a 3-pixel geometric fit window":
            raise ValueError(f"Need a numerical explanation for {record['id']}: {error}")
        # This dataset's unmeasurable contour has no excluded or repeated points.
        from steger.robust_measurement import _terminal_segment
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        assert np.all(distances > 1e-9)
        start, stop = _terminal_segment(points, normals, np.r_[0., np.cumsum(distances)])
        valid_length = float(np.linalg.norm(np.diff(points[start:stop], axis=0), axis=1).sum())
        reasons.append(
            f"中心线分段后，每段只有{valid_length * .2:.2f}像素，"
            "达不到计算方向所需的3像素。"
        )
    if coverage < .75:
        reasons.append(
            "算法找到的中心线没有贯穿mask范围的75%以上，"
            f"实际为{coverage:.2%}。"
        )
    if record["angle_deg"] is None:
        assert reasons, record["id"]
    else:
        assert not reasons, record["id"]
    return dict(reason="".join(reasons), centerline_arc_px=contour.length,
                fit_window_length_px=.2 * valid_length,
                centerline_span_px=span, mask_span_px=mask_span,
                coverage=coverage, minimum_span_px=.75 * mask_span,
                measurement_error=error)


def base(ax, roi, mask):
    ax.imshow(roi, interpolation="nearest")
    # Boundary only: no filled-mask artist is added to the image.
    ax.contour(mask.astype(float), levels=[.5], colors=[COLORS["mask"]], linewidths=1.15)
    ax.set(xticks=[], yticks=[], xlim=(-.5, roi.shape[1]-.5), ylim=(roi.shape[0]-.5, -.5))
    ax.set_aspect("equal")


def zoom(ax, mask):
    points = np.argwhere(mask)
    low, high = points.min(axis=0)-.5, points.max(axis=0)+.5
    side = min(float(np.max(high-low)+12), min(mask.shape))
    center = np.clip((low+high)/2, side/2-.5, np.asarray(mask.shape)-.5-side/2)
    ax.set(xlim=(center[1]-side/2, center[1]+side/2),
           ylim=(center[0]+side/2, center[0]-side/2))


def draw_centerline(ax, points, measurement, *, show_points=False, normals=None):
    ax.plot(points[:, 1], points[:, 0], color="#888888", ls="--", lw=1.05, zorder=4)
    start, stop = measurement.get("valid_point_start", 0), measurement.get("valid_point_stop", len(points))
    valid = points[start:stop]
    ax.plot(valid[:, 1], valid[:, 0], color=COLORS["line"], lw=1.7, zorder=5)
    if normals is not None:
        normals = normals / np.linalg.norm(normals, axis=1)[:, None]
        segments = np.stack([points-1.25*normals, points+1.25*normals], axis=1)
        ax.add_collection(LineCollection(segments[..., ::-1], colors=COLORS["normal"],
                                        linewidths=.95, alpha=.9, zorder=6))
    if show_points:
        ax.scatter(points[:, 1], points[:, 0], s=11, color=COLORS["points"],
                   edgecolors="#37543b", linewidths=.45, zorder=7)
    excluded = np.asarray(measurement.get("excluded_indices", []), dtype=int)
    if len(excluded):
        ax.scatter(points[excluded, 1], points[excluded, 0], color=COLORS["excluded"],
                   marker="x", s=48, linewidths=1.65, zorder=10)


def draw_fits(ax, points, measurement):
    draw_centerline(ax, points, measurement)
    if not measurement:
        return
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    for key, direction_key, color in (
        ("min_window_arc_px", "direction_min", COLORS["minimum"]),
        ("max_window_arc_px", "direction_max", COLORS["maximum"]),
    ):
        sample_positions = np.linspace(*measurement[key], 40)
        sample = np.column_stack([np.interp(sample_positions, arc, points[:, k]) for k in range(2)])
        center = sample.mean(axis=0)
        direction = np.asarray(measurement[direction_key])
        ax.plot(sample[:, 1], sample[:, 0], color=color, lw=3.5, alpha=.9, zorder=6)
        ends = center + np.array([[-3.8], [3.8]]) * direction
        ax.plot(ends[:, 1], ends[:, 0], color=color, ls="--", lw=1.25, zorder=7)
        ax.annotate("", xy=(center[1]+4.8*direction[1], center[0]+4.8*direction[0]),
                    xytext=(center[1], center[0]),
                    arrowprops=dict(arrowstyle="-|>", lw=1.8, color=color, mutation_scale=13), zorder=8)


def figure(record, roi, mask, timing, diagnostic, output):
    points, normals = np.asarray(record["points"]), np.asarray(record["normals"])
    measurement = record.get("measurement", {})
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 5.55))
    for ax in axes:
        base(ax, roi, mask)
    axes[0].set_title("原始ROI＋颈部mask轮廓", fontsize=12)
    zoom(axes[1], mask)
    draw_centerline(axes[1], points, measurement, show_points=True, normals=normals)
    axes[1].set_title("中心线、全部提取点与法向（放大）", fontsize=12)
    zoom(axes[2], mask)
    draw_fits(axes[2], points, measurement)
    fits_label = "局部拟合方向（放大）" if record["angle_deg"] is None else "测角拟合段与方向（放大）"
    if not measurement:
        fits_label = '小段不足3像素，未生成拟合方向'
    axes[2].set_title(fits_label, fontsize=11)
    fig.suptitle(f'{record["id"]}  |  角度：{angle_text(record["angle_deg"])}  |  平均耗时：{timing:.2f} ms',
                 fontsize=15, y=.98)
    legend = [
        Line2D([], [], color=COLORS["mask"], lw=1.5, label="mask轮廓"),
        Line2D([], [], color=COLORS["line"], lw=1.8, label="中心线"),
        Line2D([], [], marker="o", ls="", markerfacecolor=COLORS["points"], markeredgecolor="#37543b", markersize=5, label="提取点"),
        Line2D([], [], color=COLORS["normal"], lw=1.2, label="法向短线"),
        Line2D([], [], color=COLORS["minimum"], lw=2.5, label="最小拟合方向"),
        Line2D([], [], color=COLORS["maximum"], lw=2.5, label="最大拟合方向"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=6, fontsize=10,
               bbox_to_anchor=(.5, .075), frameon=False)
    note = f'共提取{len(points)}个点。蓝紫短线是法向，黄/粉箭头是拟合方向；箭头不是新增提取点。'
    if measurement.get("excluded_indices"):
        note += f' 红叉为测角排除的{len(measurement["excluded_indices"])}个原始末端点。'
    fig.text(.5, .055, note, ha="center", fontsize=9, color="#4c5965")
    if record['angle_deg'] is None:
        reason = '无最终角度：' + diagnostic['reason']
        fig.text(.5, .015, reason, ha='center', fontsize=10, color='#914212')
    fig.tight_layout(rect=(0, .18, 1, .92), w_pad=1.2)
    fig.savefig(output, dpi=165, facecolor="white")
    plt.close(fig)


def main():
    data = json.loads((SOURCE/"benchmark.json").read_text(encoding="utf-8"))
    steps = json.loads((SOURCE/"step-timing.json").read_text(encoding="utf-8"))
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == digest
               for p, digest in data["source_hashes"].items()), "Measurement source changed; remeasure first"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    images = OUTPUT/"images"
    images.mkdir(exist_ok=True)
    font = "C:/Windows/Fonts/msyh.ttc"
    fontManager.addfont(font)
    plt.rcParams.update({"font.family": FontProperties(fname=font).get_name(), "axes.unicode_minus": False})
    rows, cards, overview = [], [], []
    for record in data["results"]:
        with Image.open(record["image_path"]) as image: roi = np.asarray(image).copy()
        with Image.open(record["mask_path"]) as image: mask = np.asarray(image) != 0
        _, mask = _prepare_neck_inputs(roi, mask)
        diagnostic = numerical_diagnosis(record, mask)
        mean_ms = data["per_image"][record["id"]]["mean_ms"]
        image_path = images/f'{record["id"]}.png'
        figure(record, roi, mask, mean_ms, diagnostic, image_path)
        row = dict(id=record["id"], angle_deg=record["angle_deg"], mean_ms=mean_ms,
                   points=len(record["points"]), excluded_points=record.get("measurement", {}).get("excluded_indices", []),
                   **diagnostic)
        rows.append(row)
        caption = diagnostic["reason"]
        if record["angle_deg"] is not None:
            m = record["measurement"]
            caption = f'提取{len(record["points"])}个中心线点，测角使用{m["point_count"]}个点。'
            if m["excluded_indices"]:
                caption += f'已排除{len(m["excluded_indices"])}个方向不一致的末端点，原始点以红叉保留显示。'
        cards.append(f'<section class="result" id="{record["id"]}"><h3>{record["id"]} · '
                     f'{angle_text(record["angle_deg"])} · 平均 {mean_ms:.2f} ms</h3>'
                     f'<p>{html.escape(caption)}</p>'
                     f'<img loading="lazy" src="images/{record["id"]}.png" alt="{record["id"]}检测点、法向和拟合方向"></section>')
        overview.append((record, roi, mask, mean_ms))
        print(record["id"], angle_text(record["angle_deg"]), flush=True)
    fig, axes = plt.subplots(8, 4, figsize=(12, 23))
    for ax, (r, roi, mask, ms) in zip(axes.flat, overview, strict=False):
        base(ax, roi, mask)
        draw_fits(ax, np.asarray(r["points"]), r.get("measurement", {}))
        ax.set_title(f'{r["id"]}  {angle_text(r["angle_deg"])}\n平均 {ms:.2f} ms', fontsize=9)
    for ax in axes.flat[len(rows):]: ax.axis("off")
    fig.suptitle(TITLE+"｜29张结果总览", fontsize=17, y=.999)
    fig.tight_layout(rect=(0, 0, 1, .99), h_pad=1.1, w_pad=.7)
    fig.savefig(OUTPUT/"all-results.png", dpi=145, facecolor="white")
    plt.close(fig)

    stats = data["statistics"]
    timing = (f'平均 {stats["mean_ms"]:.2f} ms/条；中位数 {stats["median_ms"]:.2f} ms；'
              f'95%的调用在 {stats["p95_ms"]:.2f} ms 内完成；最大 {stats["max_ms"]:.2f} ms。')
    scope = (f'i5-12500，29对64×64图像，各测60次，共{stats["calls"]}次。'
             '收到ROI和mask后计时，包含找线、测角和数据交接；不含读图、分割、绘图、启动和排队，未叠加分割负载。')
    step_note = (f'单步计时另测{steps["calls"]}次，合计平均 {steps["mean_total_ms"]:.2f} ms（含计时开销）。'
                 '并行部分按每轮用时最长的任务计算，因此与整体平均值略有差别。')
    parallel_note = '先试5种平滑程度；没有找到足够完整的线时，再补试3种，所以个别图更慢。'
    coverage_note = '中心线需要贯穿mask颈部范围的至少75%；用来判断方向的小段至少要有3像素。'
    legend_note = ('左：原图与mask轮廓；中：中心线、提取点和法向；右：拟合方向。'
                   '红叉为排除点。未输出最终角度时，右图箭头仅表示局部方向。')
    short_summary = ('本批29张中，26张得到最终角度，3张未输出；具体原因见下表及对应效果图。'
                     '平均耗时接近5毫秒目标，5毫秒不是每次调用的上限。')
    failure_rows = [r for r in rows if r["angle_deg"] is None]
    step_html = ''.join(f'<tr><td>{index}</td><td>{name}</td><td>{description}</td>'
                        f'<td>{steps["stage_mean_ms"][key]:.3f}</td></tr>'
                        for index, (key, name, description) in enumerate(STEPS, 1))
    failures_html = ''.join(f'<tr><td><a href="#{r["id"]}">{r["id"]}</a></td>'
                            f'<td>{html.escape(r["reason"])}</td></tr>' for r in failure_rows)
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{TITLE}</title>
<style>body{{font:16px/1.8 system-ui,"Microsoft YaHei",sans-serif;background:#f4f7fb;color:#193248;max-width:1240px;margin:30px auto;padding:0 18px}}h1,h2,h3{{line-height:1.4}}h3{{font-size:19px}}section{{background:#fff;border-radius:12px;padding:24px;margin:22px 0}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;vertical-align:top;padding:10px;border-bottom:1px solid #dce4ec}}.muted{{font-size:14px;color:#657386}}.scroll{{overflow:auto}}img{{display:block;width:100%;height:auto}}li{{margin:10px 0}}a{{color:#1261b3}}.lead{{font-size:20px;font-weight:650}}.result{{scroll-margin-top:20px}}.result p{{margin:8px 0 16px}}@media print{{body{{max-width:none;background:white}}section{{break-inside:avoid}}}}</style>
<h1>{TITLE}</h1><p class="muted">2026-11-05 · perf/angle-worker-v1.0.1 分支基础上的当前本地版本</p>
<p>输入：精子ROI原图和对应颈部mask。输出：颈部中心线、提取点、法向、拟合方向及颈部弯曲角。</p>
<section><h2>1. 算法流程与每一步耗时</h2><div class="scroll"><table><tr><th>步骤</th><th>做什么</th><th>具体作用</th><th>平均ms/条</th></tr>{step_html}</table></div><p class="muted">{step_note}</p><p>{parallel_note}</p></section>
<section><h2>2. 角度如何计算</h2><ol>{''.join('<li>'+x+'</li>' for x in ANGLE_STEPS)}</ol><p><strong>弯曲角＝最大局部方向角－最小局部方向角；直线为0°。</strong></p><p>{coverage_note}</p></section>
<section><h2>3. 当前采用的加速手段</h2><ul>{''.join('<li><strong>'+name+'：</strong>'+text+'</li>' for name,text in ACCELERATION)}</ul><p>完整保留候选平滑程度、亮暗两种线、精细定位和测角规则。</p></section>
<section><h2>4. 当前整体耗时</h2><p class="lead">{timing}</p><p>{short_summary}</p><p class="muted">{scope}</p><table><tr><th>没有最终角度的图片</th><th>具体原因</th></tr>{failures_html}</table></section>
<section><h2>5. 每个精子的结果效果图</h2><p>{legend_note}</p></section>{''.join(cards)}</html>'''
    assert '复核' not in page
    (OUTPUT/"index.html").write_text(page, encoding="utf-8")
    md = [f'# {TITLE}', '', '输入：精子ROI原图和颈部mask；输出：中心线、提取点、法向、拟合方向和弯曲角。', '',
          '## 1. 算法流程与每一步耗时', '', '| 步骤 | 处理 | 作用 | 平均ms/条 |', '|---|---|---|---:|',
          *[f'| {i} | {name} | {text} | {steps["stage_mean_ms"][key]:.3f} |' for i,(key,name,text) in enumerate(STEPS,1)],
          '', step_note, '', parallel_note, '', '## 2. 角度如何计算', '',
          *[f'{i}. {text}' for i,text in enumerate(ANGLE_STEPS,1)], '', coverage_note, '',
          '## 3. 当前采用的加速手段', '',
          *[f'- **{name}：**{text}' for name,text in ACCELERATION], '',
          '## 4. 当前整体耗时', '', f'**{timing}**', '', scope, '', short_summary, '',
          '| 没有最终角度的图片 | 具体原因 |', '|---|---|',
          *[f'| {r["id"]} | {r["reason"]} |' for r in failure_rows], '',
          '## 5. 每个精子的结果效果图', '', legend_note, '']
    for r in rows:
        md += [f'### {r["id"]}｜{angle_text(r["angle_deg"])}｜平均 {r["mean_ms"]:.2f} ms', '',
               *([r['reason'], ''] if r['reason'] else []), f'![{r["id"]}效果图](images/{r["id"]}.png)', '']
    (OUTPUT/f'{TITLE}.md').write_text('\n'.join(md), encoding="utf-8")
    with (OUTPUT/'results.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(['图像','最终角度_度','平均耗时_ms','提取点数','排除点数','未输出最终角度的具体原因'])
        writer.writerows((r['id'],r['angle_deg'],f'{r["mean_ms"]:.6f}',r['points'],len(r['excluded_points']),r['reason']) for r in rows)
    report_data = dict(title=TITLE, timing=stats, timing_scope=scope, step_means_ms=steps['stage_mean_ms'],
                       step_mean_total_ms=steps['mean_total_ms'], results=rows,
                       source_benchmark=str(SOURCE/'benchmark.json'), source_steps=str(SOURCE/'step-timing.json'),
                       source_hashes=data['source_hashes'], generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUTPUT/'report-data.json').write_text(json.dumps(report_data, ensure_ascii=False, indent=2), encoding='utf-8')
    with zipfile.ZipFile(OUTPUT/f'{TITLE}.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUTPUT.rglob('*')):
            if path.is_file() and path.suffix != '.zip':
                archive.write(path, path.relative_to(OUTPUT))
    print(OUTPUT/'index.html')


if __name__ == '__main__':
    main()
