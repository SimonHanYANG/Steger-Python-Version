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
from scipy import ndimage

from steger import lines_gauss
from steger.io import read_image, write_json
from steger.measurement import measure_bending
from steger.visualization import save_process_report

SCALES = (1.0, 1.5, 2.0, 2.5, 3.0)
RESCUE_SCALES = (3.5, 4.0, 4.5)
LOW, HIGH = 0.05, 0.15
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


def mask_geometry(mask):
    """Use the largest 8-connected component; preserve the original for display."""
    mask = np.asarray(mask, dtype=bool)
    labels, count = ndimage.label(mask, structure=np.ones((3, 3)))
    if not count:
        raise ValueError("Empty neck mask")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    selected = labels == np.argmax(sizes)
    pixels = np.argwhere(selected)
    if len(pixels) < 6:
        raise ValueError("Neck mask contains fewer than six pixels")
    _, _, vt = np.linalg.svd(pixels - pixels.mean(axis=0), full_matrices=False)
    axis = vt[0]
    span = float(np.ptp(pixels @ axis))
    if span < 1:
        raise ValueError("Degenerate mask extent")
    distance = ndimage.distance_transform_edt(np.pad(selected, 1))[1:-1, 1:-1]
    return {
        "mask": selected,
        "axis": axis,
        "span": span,
        "distance": distance,
        "component_count": int(count),
        "ignored_mask_pixels": int(mask.sum() - selected.sum()),
        "touches_roi_border": bool(
            selected[0].any() or selected[-1].any() or selected[:, 0].any() or selected[:, -1].any()
        ),
    }


def contour_metrics(contour, geometry):
    coverage = float(np.ptp(contour.points @ geometry["axis"]) / geometry["span"])
    depth = ndimage.map_coordinates(geometry["distance"], contour.points.T, order=1)
    centrality = float(np.median(depth) / geometry["distance"].max())
    return {"span_coverage": coverage, "centrality": centrality, "length_px": contour.length}


def candidate_runs(image, geometry, scales=SCALES):
    records = []
    for sigma in scales:
        for polarity in ("dark", "light"):
            result = detect(image, geometry["mask"], sigma, polarity)
            for index, contour in enumerate(result.contours):
                if contour.closed or len(contour.points) < 6:
                    continue
                metrics = contour_metrics(contour, geometry)
                # The relaxed length is for a separately labelled LOCAL estimate only.
                try:
                    local = measure_bending(contour, min_window_length=0.5)
                except ValueError:
                    continue
                records.append(
                    {
                        "sigma": sigma,
                        "polarity": polarity,
                        "contour_index": index,
                        **metrics,
                        "local_angle_deg": local.angle_deg,
                        "points": contour.points,
                    }
                )
    return records


def detect(image, mask, sigma, polarity, *, trace=False):
    # Width/asymmetry/profile models have not been validated on these phase-like
    # real micrographs. Keep the measured grayscale ridge and expose every stage.
    return lines_gauss(
        image,
        sigma,
        LOW,
        HIGH,
        polarity,
        False,
        "none",
        False,
        mask=mask,
        profile_refinement=False,
        return_trace=trace,
    )


def choose_candidate(candidates):
    """Favor extent, then a declared reference scale; never select by angle."""
    if not candidates:
        return None
    best_coverage = max(c["span_coverage"] for c in candidates)
    eligible = [c for c in candidates if c["span_coverage"] >= best_coverage - 0.05]
    adequate = [c for c in eligible if c["span_coverage"] >= 0.75 and c["length_px"] >= 15]
    if adequate:
        eligible = adequate
    return max(
        eligible,
        key=lambda c: (-abs(c["sigma"] - 2.0), c["centrality"], c["sigma"]),
    )


def assess(contour, selected, candidates):
    local = measure_bending(contour, min_window_length=0.5)
    flags = []
    if selected["span_coverage"] < 0.75:
        flags.append("insufficient_span_coverage")
    try:
        measurement = measure_bending(contour)
    except ValueError:
        measurement = None
        flags.append("fit_windows_shorter_than_3px")
    # Extent measures projection on the mask's major axis, NOT anatomical truth.
    nearby = []
    for sigma in sorted({c["sigma"] for c in candidates}):
        options = [
            c for c in candidates if c["sigma"] == sigma and c["polarity"] == selected["polarity"]
        ]
        if not options:
            continue
        item = max(options, key=lambda c: (c["span_coverage"], c["centrality"]))
        if item["span_coverage"] >= selected["span_coverage"] - 0.08 and item["length_px"] >= 15:
            nearby.append(item["local_angle_deg"])
    scale_range = [float(min(nearby)), float(max(nearby))] if nearby else None
    if len(nearby) < 2:
        flags.append("insufficient_comparable_scales")
    elif scale_range[1] - scale_range[0] > 5:
        flags.append("scale_sensitive_over_5deg")
    window_angles = []
    for window in (0.15, 0.2, 0.25):
        try:
            m = measure_bending(contour, window_fraction=window)
            window_angles.append({"window_fraction": window, "angle_deg": m.angle_deg})
        except ValueError:
            pass
    window_range = (
        [min(v["angle_deg"] for v in window_angles), max(v["angle_deg"] for v in window_angles)]
        if window_angles
        else None
    )
    if window_range and window_range[1] - window_range[0] > 5:
        flags.append("window_sensitive_over_5deg")
    if selected["centrality"] < 0.4:
        flags.append("ridge_near_mask_boundary")
    if selected["sigma"] > max(SCALES):
        flags.append("large_scale_rescue_requires_review")
    insufficient = "insufficient_span_coverage" in flags or measurement is None
    status = "insufficient_segment" if insufficient else "review_required" if flags else "estimated"
    return {
        "status": status,
        "angle_deg": None if insufficient else measurement.angle_deg,
        "local_segment_angle_deg": local.angle_deg,
        "measurement": (measurement or local).to_dict(),
        "quality_flags": flags,
        "scale_angle_range_deg": scale_range,
        "comparable_scale_count": len(nearby),
        "window_angle_range_deg": window_range,
        "window_sensitivity": window_angles,
    }


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
    axis.plot(contour.cols, contour.rows, color="#36ff83", lw=1.3)
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
    angle_label = f"{main:.1f} deg" if main is not None else f"LOCAL ONLY: {m['angle_deg']:.1f} deg"
    fig, axs = plt.subplots(1, 3, figsize=(13, 5.1))
    _base(axs[0], image, original_mask, "Original ROI and supplied mask")
    if geometry["ignored_mask_pixels"]:
        axs[0].contour(geometry["mask"], levels=[0.5], colors=["#ffbf47"], linewidths=0.9)
    _base(axs[1], image, geometry["mask"], "Measured grayscale centerline and normals")
    _draw_axes(axs[1], contour)
    _base(axs[2], image, geometry["mask"], "Two fitted directions along the neck")
    _draw_measurement(axs[2], contour, m)
    fig.suptitle(f"{record['id']} | {angle_label} | {record['status']}", fontsize=16, y=0.98)
    fig.text(
        0.5,
        0.055,
        f"sigma={record['sigma']:g} px; {record['polarity']} ridge; "
        f"major-axis span coverage={record['span_coverage']:.0%}; "
        f"centerline length={contour.length:.1f} px\n"
        "Straight = 0 deg. Fit windows: 5-25% and 75-95% of selected line arclength. "
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
    w = record["window_sensitivity"]
    if w:
        axs[2].plot(
            [v["window_fraction"] for v in w], [v["angle_deg"] for v in w], "o-", color="#8c3eb0"
        )
    axs[2].set(
        xlabel="Fit-window arclength fraction",
        ylabel="Angle (deg)",
        title="Window sensitivity at selected scale",
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
        record.update(assess(contour, selected, candidates))
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
    counts = dict(Counter(r["status"] for r in rows))
    fields = [
        "id",
        "angle_deg",
        "local_segment_angle_deg",
        "status",
        "sigma",
        "polarity",
        "span_coverage",
        "length_px",
        "centrality",
        "cumulative_turn_diagnostic_deg",
        "scale_min_deg",
        "scale_max_deg",
        "touches_roi_border",
        "ignored_mask_pixels",
        "quality_flags",
    ]
    with (output / "angles.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            line = {key: r.get(key) for key in fields}
            line["cumulative_turn_diagnostic_deg"] = r.get("measurement", {}).get(
                "cumulative_turn_deg"
            )
            limits = r.get("scale_angle_range_deg") or (None, None)
            line.update(scale_min_deg=limits[0], scale_max_deg=limits[1])
            line["quality_flags"] = ";".join(r.get("quality_flags", []))
            writer.writerow(line)
    write_json(
        output / "results.json",
        {
            "counts": counts,
            "python": platform.python_version(),
            "method": {
                "target": "visible neck self-bending, straight=0deg; not head-neck angle",
                "angle_interpretation": "end-window direction change, not maximum local bend; S-curves can cancel; cumulative turn is a noise-sensitive diagnostic",
                "scales_px": SCALES,
                "rescue_scales_px": RESCUE_SCALES,
                "rescue_policy": "only when initial search has no contour meeting extent/length gates; every selected rescue scale requires review",
                "low": LOW,
                "high": HIGH,
                "selection": "within 0.05 of maximum PCA span coverage; prefer candidates meeting extent/length gates if available, then sigma closest to 2, then centrality",
                "mask": "largest 8-connected component; original intensities retained for convolution",
                "windows": "trim 5%, fit next 20% at each end; orthogonal line fits",
                "minimum_fit_window_length_px": 3,
                "minimum_span_coverage": 0.75,
                "scale_sensitivity_review_threshold_deg": 5,
                "width_bias_profile_fitting": "disabled: not validated on this imaging modality",
                "accuracy": "real-image ground truth and HALCON comparison unavailable; no accuracy guarantee",
                "sensitivity": "parameter variation only; NOT confidence intervals or error bounds",
            },
            "results": rows,
        },
    )
    ncols = 4
    nrows = max(1, int(np.ceil(len(display_items) / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.8 * nrows), squeeze=False)
    for ax, (record, image, mask, contour) in zip(axes.flat, display_items, strict=False):
        main = record.get("angle_deg")
        local = record.get("local_segment_angle_deg")
        value = (
            f"{main:.1f} deg"
            if main is not None
            else f"LOCAL {local:.1f} deg"
            if local is not None
            else "No result"
        )
        _base(ax, image, mask, f"{record['id']} | {value}\n{record['status']}")
        if contour is not None:
            _draw_measurement(ax, contour, record["measurement"])
    for ax in axes.flat[len(display_items) :]:
        ax.axis("off")
    fig.suptitle(
        "Visible neck self-bending | 28 supplied ROI/mask pairs"
        if len(rows) == 28
        else f"Visible neck self-bending | {len(rows)} pairs",
        fontsize=19,
        y=0.998,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output / "all-results.png", dpi=145, facecolor="white")
    plt.close(fig)

    paragraphs = [
        "# 颈部自身弯曲批量分析",
        "",
        f"共 {len(rows)} 对图像。状态统计："
        + "；".join(f"{STATUS_ZH[k]} {v} 张" for k, v in counts.items())
        + "。",
        "",
        "弯曲角定义：沿选中中心线的同一行进方向，在 5–25% 与 75–95% 弧长窗口分别进行正交直线拟合，两方向夹角，直线为 0°。不是头颈夹角，也不是最大局部折角。",
        "",
        "S 形或反向弯折可能使两端方向变化相互抵消，小角度不一定代表颈部很直。CSV 和下表另附累计转角诊断值（按约 1 像素尺度估计），该值容易受局部定位波动影响，不能直接当作高精度弯曲角或替代主角度。",
        "",
        "这是低分辨率真实灰度图上的估计，尚无人工中心线或 HALCON 对照真值。参数变化范围不是置信区间。图像触边时仅描述当前可见颈部。",
        "",
        "主角度仅在主轴投影覆盖率达到 75% 且每个拟合窗口至少 3 像素时给出；其他样本只保留局部段的诊断数值，不能代表完整颈部。75% 是本次筛查规则，不是已验证的准确度标准。",
        "",
        "初始检查 1、1.5、2、2.5、3 像素尺度；均无足够覆盖和长度时，追加 3.5、4、4.5 像素尺度，追加尺度的结果一律标记需复核，因为较强平滑可能改变细节。在最大覆盖率的 5 个百分点以内，优先保留符合覆盖和长度规则的候选（若有），再选择最接近 2 像素的高斯尺度，并检查暗线和亮线。选择过程不使用角度大小。mask 仅限定位置，卷积始终使用原始灰度。孤立 mask 小块排除但保留原图显示。",
        "",
        "本批次关闭线宽、不对称模型校正和额外灰度轮廓拟合，因为这些模型尚未在这类显微图像上验证。原始 Hessian 法向、Taylor/Newton 定位与连接过程均保存。",
        "",
        "- [全部角度 CSV](angles.csv)",
        "- [全部结构化结果和分析规则](results.json)",
        "- [全部可视化压缩包](visualizations.zip)",
        "",
        "![全部结果](all-results.png)",
        "",
        "| 图像 | 两端方向差 ° | 局部段诊断角度 ° | 累计转角诊断 ° | 状态 | 尺度变化范围 ° | 主轴投影覆盖率 | 图像触边 | 查看 |",
        "|---|---:|---:|---:|---|---|---:|---|---|",
    ]
    cards = []
    table = []
    for r in rows:
        identifier = r["id"]
        main = "—" if r.get("angle_deg") is None else f"{r['angle_deg']:.1f}"
        local = (
            "—"
            if r.get("local_segment_angle_deg") is None
            else f"{r['local_segment_angle_deg']:.1f}"
        )
        limits = r.get("scale_angle_range_deg")
        spread = "—" if limits is None else f"{limits[0]:.1f}–{limits[1]:.1f}"
        coverage = f"{r.get('span_coverage', 0):.0%}"
        border = "是" if r.get("touches_roi_border") else "否"
        state = STATUS_ZH[r["status"]]
        cumulative = r.get("measurement", {}).get("cumulative_turn_deg")
        cumulative_text = "—" if cumulative is None else f"{cumulative:.1f}"
        available = [
            (name, title)
            for name, title in (
                ("measurement.png", "测量图"),
                ("process/README.md", "中间过程"),
                ("sensitivity.png", "敏感性"),
            )
            if (output / identifier / name).exists()
        ]
        report_links = " · ".join(f"[{title}]({identifier}/{name})" for name, title in available)
        paragraphs.append(
            f"| {identifier} | {main} | {local} | {cumulative_text} | {state} | {spread} | {coverage} | {border} | {report_links} |"
        )
        table.append(
            f'<tr data-status="{r["status"]}"><td><a href="#{identifier}">{identifier}</a></td><td>{main}</td><td>{local}</td><td>{cumulative_text}</td><td>{state}</td><td>{spread}</td><td>{coverage}</td><td>{border}</td></tr>'
        )
        flags = (
            "；".join(FLAG_ZH[k] for k in r.get("quality_flags", []))
            or "未触发自动敏感性标记（不代表精度已验证）"
        )
        preview = (
            f'<img loading="lazy" src="{identifier}/measurement.png" alt="测量叠加图">'
            if (output / identifier / "measurement.png").exists()
            else ""
        )
        arrays_link = (
            f' · <a href="{identifier}/process/process.npz">导数及过程数组 NPZ</a>'
            if (output / identifier / "process/process.npz").exists()
            else ""
        )
        cards.append(
            f'<section id="{identifier}" data-status="{r["status"]}"><h2>{identifier} · {main}° · {state}</h2>'
            f"<p>局部段诊断值 {local}°；累计转角诊断 {cumulative_text}°；尺度变化 {spread}°；投影覆盖 {coverage}；触边 {border}。</p>"
            f'<p class="muted">{html.escape(flags)}</p>{preview}'
            f"<details><summary>展开法向与 Hessian 等中间过程，以及参数敏感性</summary>"
            + "".join(
                f'<a href="{identifier}/{name}"><img loading="lazy" src="{identifier}/{name}" alt="{name}"></a>'
                for name in (
                    "process/normals.png",
                    "process/process-overview.png",
                    "process/derivatives.png",
                    "process/point-inspection.png",
                    "sensitivity.png",
                )
                if (output / identifier / name).exists()
            )
            + f'</details><p><a href="{identifier}/result.json">原始坐标及测量 JSON</a>{arrays_link}</p></section>'
        )
    (output / "README.md").write_text("\n".join(paragraphs) + "\n", encoding="utf-8")
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>颈部弯曲分析</title><style>
    body{font-family:system-ui,sans-serif;margin:24px auto;padding:0 18px;max-width:1250px;color:#183047;background:#f4f7fb;line-height:1.7}h1,h2{line-height:1.3}section{background:white;padding:20px;margin:24px 0;border-radius:14px}img{display:block;width:100%;height:auto;margin:12px 0}table{width:100%;border-collapse:collapse;background:white;font-size:14px}td,th{border-bottom:1px solid #dce4ee;padding:9px;text-align:left}a{color:#185fc4}summary{cursor:pointer;font-weight:600;padding:12px 0}.muted{color:#596b7d}select{padding:8px;margin:10px 0}.scroll{overflow:auto}.note{padding:16px;background:#fff4d7;border-radius:10px}[hidden]{display:none!important}</style>
    <h1>颈部自身弯曲分析</h1>
    <p>直线为 0°；使用中心线两端内侧 20% 弧长窗口拟合，分别去除最外端 5%。测量目标为颈部自身弯曲。</p>
    <p>主角度是两端方向差，S 形弯折可能相互抵消，小角度不一定代表颈部很直。另列累计转角诊断，该值对局部波动敏感，需结合图像查看。</p>
    <p class="note">这些数值是当前可见灰度颈部段的估计，没有真值精度验证。主角度为空时，该段覆盖不足或过短；局部诊断角度不能代替完整颈部角度。尺度变化范围不是置信区间。触边图像仅描述 ROI 内可见部分。</p>
    <p><a href="angles.csv">下载角度 CSV</a> · <a href="visualizations.zip">下载全部可视化</a> · <a href="README.md">完整方法及结果说明</a> · <a href="all-results.png">查看总览大图</a></p>
    <label>显示：<select id="filter"><option value="all">全部结果</option><option value="estimated">可测估计</option><option value="review_required">需复核</option><option value="insufficient_segment">仅局部</option></select></label>
    <div class="scroll"><table><thead><tr><th>图像</th><th>两端方向差 °</th><th>局部诊断 °</th><th>累计转角诊断 °</th><th>状态</th><th>尺度变化 °</th><th>投影覆盖</th><th>触边</th></tr></thead><tbody>"""
    page += "".join(table) + "</tbody></table></div>" + "".join(cards)
    page += """<script>document.querySelector('#filter').addEventListener('change',e=>{document.querySelectorAll('[data-status]').forEach(el=>el.hidden=e.target.value!=='all'&&el.dataset.status!==e.target.value)});</script></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
    with zipfile.ZipFile(
        output / "visualizations.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
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
