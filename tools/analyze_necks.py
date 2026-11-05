"""Analyze paired neck ROIs/masks and export auditable local batch reports.

This script measures the visible grayscale ridge, not the head-neck angle.
It never invents a missing neck segment or substitutes a binary skeleton.
Requires the visualization extra. Real input images and their reports stay local.
"""

import argparse
import csv
import hashlib
import html
import json
import platform
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from steger.io import read_image, write_json
from steger.neck import (
    HIGH,
    LOW,
    RESCUE_SCALES,
    SCALES,
    assess,
    candidate_runs,
    choose_candidate,
    detect,
    mask_geometry,
)
from steger.neck import (
    contour_metrics as contour_metrics,
)
from steger.visualization import save_process_report

STATUS_ZH = {
    "estimated": "可测估计",
    "review_required": "需复核",
    "insufficient_segment": "覆盖不足或过短",
    "no_line": "无可测中心线",
    "input_error": "输入错误",
}
FLAG_ZH = {
    "insufficient_span_coverage": "中心线主轴投影覆盖不足 75%",
    "fit_windows_shorter_than_3px": "端部拟合窗口不足 3 像素",
    "insufficient_direction_samples": "可用中心线过短或未通过几何质量检查",
    "terminal_samples_excluded": "末端方向不一致，已标记排除点",
    "terminal_direction_requires_review": "末端方向与法向存在较大差异",
    "nonlinear_fit_window": "局部窗口偏离直线较大",
    "abrupt_local_direction_change": "相邻拟合方向变化过大",
    "duplicate_samples_ignored": "已忽略重复坐标",
    "insufficient_trusted_span_coverage": "排除异常末端后覆盖不足75%",
    "insufficient_comparable_scales": "具有相近覆盖范围的可比较尺度不足两个",
    "scale_sensitive_over_5deg": "尺度变化导致角度差超过 5°",
    "window_sensitive_over_5deg": "拟合窗口变化导致角度差超过 5°",
    "ridge_near_mask_boundary": "提取线靠近 mask 边界，可能不是颈部中央",
    "large_scale_rescue_requires_review": "使用较大平滑尺度才获得足够覆盖，需检查细节损失",
}


def image_pairs(directory):
    """Validate pairing instead of silently dropping an unmatched mask or ROI."""
    root = Path(directory)
    images = sorted(
        p for p in root.iterdir() if p.suffix.lower() == ".png" and not p.stem.endswith("_mask")
    )
    masks = {p.stem[:-5]: p for p in root.glob("*_mask.png")}
    missing = [p.name for p in images if p.stem not in masks]
    orphan = sorted(set(masks) - {p.stem for p in images})
    if missing or orphan:
        raise ValueError(f"Unpaired input: images without masks={missing}; orphan masks={orphan}")
    if not images:
        raise ValueError(f"No PNG image/mask pairs in {root}")
    return [(p, masks[p.stem]) for p in images]


def _base(axis, image, mask, title):
    axis.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    axis.contour(mask, levels=[0.5], colors=["#20d8ff"], linewidths=0.75)
    axis.set(xlim=(-0.5, image.shape[1] - 0.5), ylim=(image.shape[0] - 0.5, -0.5))
    axis.set_title(title, fontsize=10)
    axis.set(xticks=[], yticks=[])


def _sample(contour, q):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(contour.points, axis=0), axis=1))]
    return np.column_stack([np.interp(q, s, contour.points[:, k]) for k in range(2)])


def _draw_measurement(axis, contour, measurement):
    if measurement.get('method') == 'geometry_arclength_direction_range_v2':
        p=contour.points
        axis.plot(p[:,1],p[:,0],color='#aaaaaa',lw=1)
        lo,hi=measurement['valid_point_start'],measurement['valid_point_stop']
        axis.plot(p[lo:hi,1],p[lo:hi,0],color='#36ff83',lw=1.3)
        excluded=np.asarray(measurement['excluded_indices'],dtype=int)
        if len(excluded):
            axis.scatter(p[excluded,1],p[excluded,0],marker='x',s=26,c='#e54435',zorder=8)
        for window,key,color in [('min_window_arc_px','direction_min','#ffbf47'),
                                  ('max_window_arc_px','direction_max','#ff6be5')]:
            samples=_sample(contour,np.linspace(*measurement[window],40))
            center=samples.mean(axis=0);v=measurement[key]
            axis.plot(samples[:,1],samples[:,0],color=color,lw=3)
            axis.quiver(center[1],center[0],v[1]*7,v[0]*7,
                        angles='xy',scale_units='xy',scale=1,color=color,width=.008)
        return
    axis.plot(contour.cols, contour.rows, color="#36ff83", lw=1.3)
    if measurement.get("method", "").startswith("cpp_legacy_"):
        for window, angle, color in (
            ("min_window", "min_angle_rad", "#ffbf47"),
            ("max_window", "max_angle_rad", "#ff6be5"),
        ):
            lo, hi = measurement[window]
            p = contour.points[lo:hi]
            center = p.mean(axis=0)
            theta = measurement[angle]
            axis.plot(p[:, 1], p[:, 0], color=color, lw=3)
            axis.quiver(center[1], center[0], 7*np.cos(theta), -7*np.sin(theta),
                        angles="xy", scale_units="xy", scale=1, color=color, width=.008)
        return
    if not measurement:
        return
    for key, direction, color in (
        ("start_window", "direction_start", "#ffbf47"),
        ("end_window", "direction_end", "#ff6be5"),
    ):
        lo, hi = measurement[key]
        points = _sample(contour, np.linspace(lo, hi, 40))
        center = points.mean(axis=0)
        v = np.asarray(measurement[direction])
        axis.plot(points[:, 1], points[:, 0], color=color, lw=3)
        axis.quiver(
            center[1],
            center[0],
            v[1] * 7,
            v[0] * 7,
            angles="xy",
            scale_units="xy",
            scale=1,
            color=color,
            width=0.008,
        )


def _draw_axes(axis, contour):
    step = max(1, int(np.ceil(len(contour.points) / 16)))
    p, n = contour.points[::step], contour.normals[::step]
    segments = np.stack([p - 2 * n, p + 2 * n], axis=1)
    axis.add_collection(LineCollection(segments[..., ::-1], colors="#20d8ff", linewidths=1))
    axis.plot(contour.cols, contour.rows, color="#36ff83", lw=1)


def save_measurement_figures(image, original_mask, geometry, contour, candidates, record, output):
    m = record["measurement"]
    main = record["angle_deg"]
    angle_label = f"{main:.1f} deg" if main is not None else "No complete-neck angle"
    fig, axs = plt.subplots(1, 3, figsize=(13, 5.1))
    _base(axs[0], image, original_mask, "Original ROI and supplied mask")
    if geometry["ignored_mask_pixels"]:
        axs[0].contour(geometry["mask"], levels=[0.5], colors=["#ffbf47"], linewidths=0.9)
    _base(axs[1], image, geometry["mask"], "Measured grayscale centerline and normals")
    _draw_axes(axs[1], contour)
    _base(axs[2], image, geometry["mask"], "Minimum / maximum local fitted directions")
    _draw_measurement(axs[2], contour, m)
    fig.suptitle(f"{record['id']} | {angle_label} | {record['status']}", fontsize=16, y=0.98)
    fig.text(
        0.5,
        0.055,
        f"sigma={record['sigma']:g} px; {record['polarity']} ridge; "
        f"major-axis span coverage={record['span_coverage']:.0%}; "
        f"centerline length={contour.length:.1f} px\n"
        "Geometry v2: arc-length local fits; continuous directions; validated tips. "
        "Cropped ROIs describe the visible segment only.",
        ha="center",
        fontsize=9,
        linespacing=1.7,
    )
    fig.subplots_adjust(left=0.025, right=0.98, top=0.85, bottom=0.19, wspace=0.08)
    fig.savefig(output / "measurement.png", dpi=160, facecolor="white")
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(13, 4.9))
    _base(axs[0], image, geometry["mask"], "Best extent candidate at each scale (same polarity)")
    all_angles = []
    tested_scales = sorted({c["sigma"] for c in candidates})
    for scale_index, sigma in enumerate(tested_scales):
        color = plt.get_cmap("tab10")(scale_index)
        options = [
            c for c in candidates if c["sigma"] == sigma and c["polarity"] == record["polarity"]
        ]
        if not options:
            continue
        c = max(options, key=lambda x: (x["span_coverage"], x["centrality"]))
        p = c["points"]
        axs[0].plot(p[:, 1], p[:, 0], color=color, lw=0.9, label=f"{sigma:g}")
        if c["local_angle_deg"] is not None:
            all_angles.append((sigma, c["local_angle_deg"], c["span_coverage"]))
    axs[0].legend(title="sigma (px)", fontsize=7, title_fontsize=8)
    if all_angles:
        a = np.asarray(all_angles)
        axs[1].plot(a[:, 0], a[:, 1], "o-", color="#245bd5")
        for sigma, angle, coverage in all_angles:
            axs[1].annotate(
                f"{coverage:.0%}",
                (sigma, angle),
                xytext=(0, 7),
                textcoords="offset points",
                fontsize=8,
            )
    axs[1].set(
        xlabel="Gaussian scale sigma (px)",
        ylabel="Local segment angle (deg)",
        title="Scale sensitivity (labels = span coverage)",
    )
    axs[1].margins(y=0.25)
    smooth = record.get("measurement", {}).get("direction_angles_rad", [])
    if smooth:
        axs[2].plot(np.rad2deg(smooth), color="#8c3eb0")
    axs[2].set(
        xlabel="Local fit window index",
        ylabel="Continuous tangent angle (deg)",
        title="Geometric direction sequence",
    )
    for ax in axs[1:]:
        ax.grid(alpha=0.15)
    fig.suptitle(f"{record['id']} | sensitivity diagnostics", fontsize=16, y=0.98)
    fig.text(
        0.5,
        0.035,
        "These ranges are sensitivity checks, NOT confidence intervals or accuracy bounds.\nDifferent scales can locate different visible segments; inspect extent and overlays together.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(left=0.025, right=0.98, top=0.82, bottom=0.22, wspace=0.30)
    fig.savefig(output / "sensitivity.png", dpi=160, facecolor="white")
    plt.close(fig)


def analyze_pair(image_path, mask_path, directory):
    image = read_image(image_path)
    supplied_mask = read_image(mask_path, mask=True)
    if image.shape != supplied_mask.shape:
        raise ValueError("ROI and mask sizes differ")
    if np.any(image < 0) or np.any(image > 255):
        raise ValueError("This batch policy expects the original 0-255 intensity scale")
    geometry = mask_geometry(supplied_mask)
    candidates = candidate_runs(image, geometry)
    tested_scales = SCALES
    if not any(c["span_coverage"] >= 0.75 and c["length_px"] >= 15 for c in candidates):
        candidates.extend(candidate_runs(image, geometry, RESCUE_SCALES))
        tested_scales += RESCUE_SCALES
    selected = choose_candidate(candidates)
    record = {
        "id": image_path.stem,
        "image_path": str(image_path.resolve()),
        "mask_path": str(mask_path.resolve()),
        "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
        "shape": list(image.shape),
        "tested_scales_px": list(tested_scales),
        **{
            k: geometry[k] for k in ("component_count", "ignored_mask_pixels", "touches_roi_border")
        },
    }
    if selected is None:
        record.update(status="no_line", angle_deg=None, local_segment_angle_deg=None)
        result = detect(image, geometry["mask"], 2.0, "dark", trace=True)
        save_process_report(
            result, directory / "process", label=f"{image_path.stem}: no measurable line"
        )
    else:
        result = detect(
            image, geometry["mask"], selected["sigma"], selected["polarity"], trace=True
        )
        contour = result.contours[selected["contour_index"]]
        record.update({k: v for k, v in selected.items() if k not in ("points", "local_angle_deg")})
        record.update(assess(contour, selected, candidates, geometry))
        record["candidate_search"] = [
            {k: v for k, v in c.items() if k != "points"} for c in candidates
        ]
        record["boundary_scope"] = (
            "visible cropped segment only"
            if geometry["touches_roi_border"]
            else "supplied neck ROI"
        )
        save_process_report(
            result,
            directory / "process",
            inspect_point=contour.points[len(contour.points) // 2],
            label=f"{image_path.stem}: real grayscale ROI; selected contour {selected['contour_index']}",
        )
        save_measurement_figures(
            image, supplied_mask, geometry, contour, candidates, record, directory
        )
    write_json(directory / "result.json", {**record, "detector": result.to_dict()})
    return (
        record,
        image,
        supplied_mask,
        result.contours[selected["contour_index"]] if selected else None,
    )


def write_summary(rows, display_items, output):
    """Concise report for the geometric bending measurement."""
    counts = dict(Counter(r["status"] for r in rows))
    fields = ["id", "angle_deg", "local_segment_angle_deg", "status", "sigma", "polarity",
              "span_coverage", "length_px", "centrality", "touches_roi_border", "quality_flags"]
    with (output / "angles.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            line = {key: r.get(key) for key in fields}
            line["quality_flags"] = ";".join(r.get("quality_flags", []))
            writer.writerow(line)
    method = {
        "target": "visible neck self-bending; not head-neck alignment",
        "angle_interpretation": "range of continuous geometric local TLS directions",
        "window_fraction": .2, "trim_fraction": .05,
        "angle_arithmetic": "float64; arc-length resampling; batched orthogonal fits",
        "angle_unwrap": True, "minimum_window_length_px": 3,
        "temporal_median": "separate NeckBendingHistory instance per tracked sperm",
        "scales_px": SCALES, "rescue_scales_px": RESCUE_SCALES,
        "low": LOW, "high": HIGH, "minimum_span_coverage": .75,
        "mask": "largest 8-connected component; convolution retains original image",
        "note": "same angle rule does not establish Python/HALCON centerline equivalence",
    }
    write_json(output / "results.json", dict(counts=counts, python=platform.python_version(),
                                              method=method, results=rows))
    ncols = 4
    nrows = max(1, int(np.ceil(len(display_items) / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.8*nrows), squeeze=False)
    for ax, (record, image, mask, contour) in zip(axes.flat, display_items, strict=False):
        main = record.get("angle_deg")
        text = "No full angle" if main is None else f"{main:.2f} deg"
        _base(ax, image, mask, f"{record['id']} | {text}\n{record['status']}")
        if contour is not None:
            _draw_measurement(ax, contour, record.get("measurement", {}))
    for ax in axes.flat[len(display_items):]:
        ax.axis("off")
    fig.suptitle("Neck bending | geometric local direction range v2", fontsize=18, y=.998)
    fig.tight_layout(rect=(0, 0, 1, .985))
    fig.savefig(output / "all-results.png", dpi=145, facecolor="white")
    plt.close(fig)
    paragraphs = ["# 几何方向范围颈部弯曲角", "",
        "中心线端点质量检查 → 等弧长局部拟合 → 连续方向最大值减最小值。覆盖整条有效中心线的5–95%，窗口长为20%弧长，直线为0°。",
        "", "拟合窗口不足3px或有效覆盖不足75%时不返回完整角度。红叉表示被排除的原始末端点，原始提取坐标保留。",
        "", f"共{len(rows)}张：{counts}。", "",
        "[全部角度CSV](angles.csv) · [结构化结果](results.json) · [可视化ZIP](visualizations.zip)", "",
        "| 图像 | 角度 ° | 状态 | 查看 |", "|---|---:|---|---|"]
    cards = []
    for r in rows:
        identifier = r["id"]
        angle = "—" if r.get("angle_deg") is None else f"{r['angle_deg']:.2f}"
        state = STATUS_ZH[r["status"]]
        figure = output / identifier / "measurement.png"
        link = f"[{identifier}]({identifier}/measurement.png)" if figure.exists() else "—"
        paragraphs.append(f"| {identifier} | {angle} | {state} | {link} |")
        image = (f'<img loading="lazy" src="{identifier}/measurement.png" alt="{identifier}">'
                 if figure.exists() else "")
        flags = "；".join(FLAG_ZH.get(k, k) for k in r.get("quality_flags", []))
        cards.append(f'<section id="{identifier}"><h2>{identifier} · {angle}° · {state}</h2>'
                     f'<p>{html.escape(flags)}</p>{image}</section>')
    (output / "README.md").write_text("\n".join(paragraphs)+"\n", encoding="utf-8")
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>几何方向范围颈部弯曲角</title>
    <style>body{font:16px/1.7 system-ui;max-width:1200px;margin:30px auto;padding:0 20px;color:#183047}section{padding:18px;border-bottom:1px solid #ddd}img{max-width:100%;height:auto}a{color:#185fc4}</style>
    <h1>几何方向范围颈部弯曲角</h1><p>有效中心线5–95%范围内，以20%弧长窗口逐段拟合方向，展开周期后取方向范围。黄色与粉色表示最小、最大方向窗口，红叉表示被排除的末端点。</p>
    <p>拟合窗口不足3px或有效覆盖不足75%时不返回完整角度。此报告是单帧结果，不混合不同精子的历史。</p>
    <p><a href="angles.csv">角度CSV</a> · <a href="results.json">结构化结果</a> · <a href="all-results.png">总览图</a></p>'''
    (output / "index.html").write_text(page+"".join(cards)+"</html>", encoding="utf-8")
    with zipfile.ZipFile(output / "visualizations.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*.png")):
            archive.write(path, path.relative_to(output))
    return counts



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/neck-analysis/results"))
    args = parser.parse_args(argv)
    pairs = image_pairs(args.directory)
    args.output.mkdir(parents=True, exist_ok=True)
    rows, display_items = [], []
    for image_path, mask_path in pairs:
        directory = args.output / image_path.stem
        directory.mkdir(parents=True, exist_ok=True)
        try:
            record, image, mask, contour = analyze_pair(image_path, mask_path, directory)
            display_items.append((record, image, mask, contour))
        except (ValueError, OSError) as error:
            record = {
                "id": image_path.stem,
                "status": "input_error",
                "angle_deg": None,
                "error": str(error),
            }
            write_json(directory / "result.json", record)
        rows.append(record)
        print(
            f"{len(rows)}/{len(pairs)} {record['id']} angle={record.get('angle_deg')} status={record['status']}",
            flush=True,
        )
    counts = write_summary(rows, display_items, args.output)
    print(json.dumps(counts, ensure_ascii=False))
    print((args.output / "index.html").resolve())
    return 1 if counts.get("input_error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
