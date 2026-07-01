"""
从训练/验证集提取 label 子图，用于训练 box-only 检测器.
阶段二：在 label crop 内只检测 box (class_id=0).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from tqdm import tqdm
from utils.logger import get_logger

logger = get_logger(__name__)

# 映射: 原 class_id → 阶段二 class_id (只有 box → 0)
CLASS_MAP = {9: 0}  # box → 0


def extract_label_crops(
    image_dir: str,
    label_dir: str,
    output_image_dir: str,
    output_label_dir: str,
    min_crop_size: int = 32,
    margin: float = 0.05,  # 5% padding around label
):
    """
    对每张图找到 label bbox → crop → 重新映射 box 坐标到 crop 空间.
    """
    img_path = Path(image_dir)
    lbl_path = Path(label_dir)
    out_img = Path(output_image_dir)
    out_lbl = Path(output_label_dir)
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    total = 0
    has_box = 0
    skipped = 0

    for lbl_file in tqdm(sorted(lbl_path.glob("*.txt")), desc="Extracting crops"):
        stem = lbl_file.stem

        # 读取标签
        classes, bboxes = [], []
        with open(lbl_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    classes.append(int(float(parts[0])))
                    bboxes.append([float(x) for x in parts[1:5]])

        # 找 label bbox (class 8) — 取置信度最高/最大的
        label_indices = [i for i, c in enumerate(classes) if c == 8]
        if not label_indices:
            skipped += 1
            continue

        # 如果多个 label，取面积最大的
        best_idx = label_indices[0]
        best_area = 0
        for idx in label_indices:
            _, _, w, h = bboxes[idx]
            area = w * h
            if area > best_area:
                best_area = area
                best_idx = idx

        lx, ly, lw, lh = bboxes[best_idx]  # YOLO: xc, yc, w, h (normalized)

        # 找图片
        img_path_candidate = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            candidate = img_path / f"{stem}{ext}"
            if candidate.exists():
                img_path_candidate = candidate
                break
        if img_path_candidate is None:
            skipped += 1
            continue

        image = cv2.imread(str(img_path_candidate))
        if image is None:
            skipped += 1
            continue

        h_img, w_img = image.shape[:2]

        # YOLO normalized → pixel coordinates
        x1 = int((lx - lw / 2) * w_img)
        y1 = int((ly - lh / 2) * h_img)
        x2 = int((lx + lw / 2) * w_img)
        y2 = int((ly + lh / 2) * h_img)

        # Add margin
        mw = int((x2 - x1) * margin)
        mh = int((y2 - y1) * margin)
        x1 = max(0, x1 - mw)
        y1 = max(0, y1 - mh)
        x2 = min(w_img, x2 + mw)
        y2 = min(h_img, y2 + mh)

        if x2 - x1 < min_crop_size or y2 - y1 < min_crop_size:
            skipped += 1
            continue

        # Crop
        crop = image[y1:y2, x1:x2]
        crop_h, crop_w = crop.shape[:2]

        # 重新映射 box bboxes 到 crop 空间
        new_bboxes = []
        new_classes = []
        for ci, bbox in zip(classes, bboxes):
            if ci not in CLASS_MAP:
                continue  # 只保留 box
            bx, by, bw, bh = bbox  # YOLO normalized in full image

            # Full image pixel coords
            bx1 = (bx - bw / 2) * w_img
            by1 = (by - bh / 2) * h_img
            bx2 = (bx + bw / 2) * w_img
            by2 = (by + bh / 2) * h_img

            # Shift to crop coords
            bx1_c = bx1 - x1
            by1_c = by1 - y1
            bx2_c = bx2 - x1
            by2_c = by2 - y1

            # Clamp to crop bounds
            bx1_c = max(0, min(crop_w, bx1_c))
            by1_c = max(0, min(crop_h, by1_c))
            bx2_c = max(0, min(crop_w, bx2_c))
            by2_c = max(0, min(crop_h, by2_c))

            # Filter invalid boxes
            if bx2_c - bx1_c < 2 or by2_c - by1_c < 2:
                continue

            # Convert back to YOLO normalized (in crop space)
            nxc = ((bx1_c + bx2_c) / 2) / crop_w
            nyc = ((by1_c + by2_c) / 2) / crop_h
            nw = (bx2_c - bx1_c) / crop_w
            nh = (by2_c - by1_c) / crop_h

            new_bboxes.append([nxc, nyc, nw, nh])
            new_classes.append(CLASS_MAP[ci])

        # 保存 crop
        out_img_path = out_img / f"{stem}.jpg"
        cv2.imwrite(str(out_img_path), crop)

        # 保存 label
        out_lbl_path = out_lbl / f"{stem}.txt"
        with open(out_lbl_path, "w") as f:
            for ci, bbox in zip(new_classes, new_bboxes):
                f.write(f"{ci} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

        total += 1
        if new_classes:
            has_box += 1

    logger.info(f"Extracted {total} crops ({has_box} with box), skipped {skipped}")
    return total, has_box


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract label crops for box-only training")
    parser.add_argument("--image_dir", default="./data/images/train")
    parser.add_argument("--label_dir", default="./data/labels/train")
    parser.add_argument("--output_image_dir", default="./data/box_crops/images/train")
    parser.add_argument("--output_label_dir", default="./data/box_crops/labels/train")
    parser.add_argument("--margin", type=float, default=0.05)
    args = parser.parse_args()

    extract_label_crops(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        output_image_dir=args.output_image_dir,
        output_label_dir=args.output_label_dir,
        margin=args.margin,
    )
