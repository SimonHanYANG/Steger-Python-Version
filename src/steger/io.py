"""Lossless numeric inputs and explicit JSON/overlay outputs."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def read_image(path, *, mask=False):
    path = Path(path)
    a = (
        np.load(path, allow_pickle=False)
        if path.suffix.lower() == ".npy"
        else np.asarray(Image.open(path))
    )
    if mask:
        if a.ndim == 3:
            a = np.any(a[..., :3] != 0, axis=-1)
        return a != 0
    if a.ndim == 3:
        if a.shape[-1] < 3:
            raise ValueError("Unsupported image channels")
        a = np.asarray(a[..., :3], dtype=np.float64) @ np.array([0.299, 0.587, 0.114])
    return a


def _safe(value):
    if isinstance(value, dict):
        return {k: _safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    return value


def write_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def overlay(image, result, path, *, scale=3):
    a = np.asarray(image, dtype=float)
    lo, hi = np.percentile(a, [0.5, 99.5])
    gray = np.clip((a - lo) / max(hi - lo, 1e-12) * 255, 0, 255).astype(np.uint8)
    canvas = Image.fromarray(gray).convert("RGB").resize((a.shape[1] * scale, a.shape[0] * scale))
    draw = ImageDraw.Draw(canvas)
    for i, c in enumerate(result.contours):
        xy = [((float(col) + 0.5) * scale, (float(row) + 0.5) * scale) for row, col in c.points]
        if len(xy) > 1:
            draw.line(xy, fill=(40, 255, 80), width=1)
        if xy:
            draw.text(xy[len(xy) // 2], str(i), fill=(255, 220, 0))
    for j in result.junctions:
        y, x = (j.point + 0.5) * scale
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), outline=(255, 80, 80))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(p)
