# -*- coding: utf-8 -*-
"""渲染原图与多组后端预测；固定类别颜色和裁块坐标，不读取 GT。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# OpenCV 使用 BGR；所有图片、所有模型共用同一图例。
COLORS = {1: (45, 185, 245), 0: (230, 145, 60)}
BOUNDARY_COLOR = (220, 35, 210)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def read_image(path, flags=cv2.IMREAD_COLOR):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def write_png(path, image):
    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Cannot encode PNG: {path}")
    data.tofile(path)


def _unique_json(pairs):
    values = {}
    for key, value in pairs:
        if key in values:
            raise ValueError(f"Duplicate class-map key: {key}")
        values[key] = value
    return values


def read_prediction(directory, stem, shape):
    mask_path = directory / f"{stem}_inst.png"
    class_path = directory / f"{stem}_class.json"
    # 除了解码结果，还核查文件本身是16位灰度PNG。
    with mask_path.open("rb") as stream:
        header = stream.read(29)
    if (len(header) < 29 or header[:8] != b"\x89PNG\r\n\x1a\n"
            or header[12:16] != b"IHDR" or header[24:26] != bytes((16, 0))):
        raise ValueError(f"Expected genuine 16-bit grayscale PNG: {mask_path}")
    instances = read_image(mask_path, cv2.IMREAD_UNCHANGED)
    if instances.ndim != 2 or instances.dtype != np.uint16 or instances.shape != tuple(shape):
        raise ValueError(f"Instance shape/dtype mismatch: {mask_path}: {instances.shape}, {instances.dtype}")
    raw = json.loads(class_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected an instance-to-class object: {class_path}")
    classes = {}
    for key, value in raw.items():
        instance_id = int(key)
        if str(instance_id) != key or not 1 <= instance_id <= 65535:
            raise ValueError(f"Invalid instance ID {key!r}: {class_path}")
        if type(value) is not int or value not in (0, 1):
            raise ValueError(f"Class must be integer 0 or 1, got {value!r}: {class_path}")
        classes[instance_id] = value
    present = {int(value) for value in np.unique(instances) if value != 0}
    if present != set(classes):
        raise ValueError(f"Mask/class IDs disagree: {stem}: mask-only={present - set(classes)}, "
                         f"class-only={set(classes) - present}")
    counts = {"F": sum(value == 1 for value in classes.values()),
              "P": sum(value == 0 for value in classes.values())}
    return instances, classes, counts


def overlay(image, instances, classes):
    result = image.copy()
    lookup = np.full(65536, -1, dtype=np.int8)
    for instance_id, value in classes.items():
        lookup[instance_id] = value
    semantic = lookup[instances]
    for value, color in COLORS.items():
        selected = semantic == value
        result[selected] = np.rint(0.78 * image[selected] + 0.22 * np.array(color)).astype(np.uint8)
    boundary = np.zeros(instances.shape, dtype=bool)
    # 在显示网格计算边界，保证缩略后仍可见；每组使用同一缩放与线宽。
    boundary[:, 1:] |= instances[:, 1:] != instances[:, :-1]
    boundary[1:, :] |= instances[1:, :] != instances[:-1, :]
    result[boundary] = BOUNDARY_COLOR
    return result


def text_line(image, text, xy, width, size=0.52, color=(35, 35, 35)):
    text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, 1)[0][0]
    scale = size * min(1.0, max(1, width - 16) / max(text_width, 1))
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def comparison(stem, image, predictions, *, width=420, height=350, crop=None):
    if crop is not None:
        x, y, side = crop
        image = image[y:y + side, x:x + side]
        predictions = [(title, mask[y:y + side, x:x + side], classes, counts)
                       for title, mask, classes, counts in predictions]
    scale = min(width / image.shape[1], height / image.shape[0])
    size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    display = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    panels = []
    rows = [("Original", display, None)]
    for title, mask, classes, counts in predictions:
        small_mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        rows.append((title, overlay(display, small_mask, classes), counts))
    for title, body, counts in rows:
        panel = np.full((height + 52, width, 3), 248, dtype=np.uint8)
        top, left = 52 + (height - size[1]) // 2, (width - size[0]) // 2
        panel[top:top + size[1], left:left + size[0]] = body
        text_line(panel, title, (8, 20), width)
        detail = stem if counts is None else f"Full-image counts: F {counts['F']} / P {counts['P']}"
        text_line(panel, detail, (8, 40), width, size=0.43)
        panels.append(panel)
    row = np.concatenate(panels, axis=1)
    legend = np.full((36, row.shape[1], 3), 255, dtype=np.uint8)
    note = "F: ferrite (gold) | P: pearlite (blue) | Magenta: instance boundaries | Counts are descriptive, not scores"
    if crop is not None:
        note = f"{stem} | Native crop x={crop[0]}, y={crop[1]}, size={crop[2]} | " + note
    text_line(legend, note, (8, 24), row.shape[1], size=0.42)
    return np.concatenate((row, legend), axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--prediction", action="append", required=True, metavar="TITLE=DIRECTORY")
    parser.add_argument("--images", nargs="+", default=["test_101", "test_091", "test_130", "test_089"])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    groups = []
    for spec in args.prediction:
        title, separator, directory = spec.partition("=")
        if not separator or not title.strip() or not title.isascii() or any(c in title for c in "\r\n"):
            raise ValueError("--prediction must be an English TITLE=DIRECTORY")
        groups.append((title.strip(), Path(directory)))
    if len({title for title, _ in groups}) != len(groups):
        raise ValueError("Prediction titles must be distinct")
    image_dir, output_dir = Path(args.image_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    available = [path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES]
    names = [Path(name).stem for name in args.images]
    if len(set(names)) != len(names):
        raise ValueError("Requested image names must be distinct")
    overview, records = [], []
    for stem in names:
        matches = [path for path in available if path.stem == stem]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one source image for {stem}, found {len(matches)}")
        image = read_image(matches[0])
        predictions = [(title, *read_prediction(directory, stem, image.shape[:2])) for title, directory in groups]
        write_png(output_dir / f"{stem}_comparison.png", comparison(stem, image, predictions))
        overview.append(comparison(stem, image, predictions, width=256, height=214))
        record = {"image": str(matches[0]), "counts": {title: counts for title, _, _, counts in predictions}}
        if stem == "test_101":
            h, w = image.shape[:2]
            if min(h, w) < 256:
                raise ValueError("test_101 must be at least 256 pixels on both axes for native crops")
            y = max(0, min(h - 256, h // 2 - 128))
            crops = {}
            for name, fraction in (("left", 0.25), ("right", 0.75)):
                x = max(0, min(w - 256, round(w * fraction) - 128))
                crop = (x, y, 256)
                crops[name] = list(crop)
                write_png(output_dir / f"{stem}_{name}.png",
                          comparison(stem, image, predictions, width=256, height=256, crop=crop))
            record["native_crops_xy_size"] = crops
        records.append(record)
    write_png(output_dir / "overview.png", np.concatenate(overview, axis=0))
    manifest = {"groups": [{"title": title, "directory": str(directory)} for title, directory in groups],
                "color_bgr": {"ferrite": COLORS[1], "pearlite": COLORS[0], "boundary": BOUNDARY_COLOR},
                "fill_alpha": 0.22, "ground_truth_used": False,
                "note": "Fixed image coordinates and colors; counts are descriptive only", "images": records}
    (output_dir / "render_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"images": len(records), "groups": len(groups), "output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
