"""
数据过采样脚本 — 针对小样本类别（damage-6, wrinkle-7, box-9）做数据增强.
box(9) 仅 147 张原始训练图（518 实例），是最大的性能瓶颈（AP50=0.475）。
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import shutil
from typing import Set

import albumentations as A
import cv2
import numpy as np
from tqdm import tqdm
from utils.logger import get_logger

logger = get_logger(__name__)

# 需要过采样的类别（damage, wrinkle, box）— box(9) 仅 147 张原始图，AP50=0.475
LOW_SAMPLE_IDS: Set[int] = {6, 7, 9}
# 每类的额外倍数: damage×3, wrinkle×2, box×3 (oversample_factor=4 → box 每图 12 副本)
CLASS_MULTIPLIER = {6: 3, 7: 2, 9: 3}


def _clamp_bboxes_yolo(bboxes, eps=1e-6):
    """
    将 YOLO bbox [xc, yc, w, h] 转为 corner [x1,y1,x2,y2] 后 clamp 到 [0,1]，
    再转回 YOLO 格式。这是唯一能保证所有角点都在合法范围内的方式。
    """
    clamped = []
    for b in bboxes:
        xc, yc, w, h = b[0], b[1], b[2], b[3]
        x1 = xc - w / 2.0
        y1 = yc - h / 2.0
        x2 = xc + w / 2.0
        y2 = yc + h / 2.0
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))
        # 如果 bbox 被 clamp 到面积为 0，微调保证最小尺寸
        if x2 - x1 < eps:
            x2 = min(1.0, x1 + eps)
            x1 = max(0.0, x2 - eps)
        if y2 - y1 < eps:
            y2 = min(1.0, y1 + eps)
            y1 = max(0.0, y2 - eps)
        new_xc = (x1 + x2) / 2.0
        new_yc = (y1 + y2) / 2.0
        new_w = x2 - x1
        new_h = y2 - y1
        clamped.append([new_xc, new_yc, new_w, new_h])
    return clamped


def _get_augmentation_pipeline(box_safe: bool = False) -> A.Compose:
    """构建数据增强管线，保持 bbox 坐标同步变换.

    Args:
        box_safe: True 时（图中含 box）禁用几何变形，保护 box bbox。
    """
    if box_safe:
        # 含 box 的图：只用颜色/光照增强，禁用几何变形
        transforms = [
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=1.0),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=30, p=0.7),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.HorizontalFlip(p=0.5),
        ]
    else:
        transforms = [
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.8),
            A.GaussNoise(p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.4),
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.8, 1.2), translate_percent=(-0.05, 0.05), rotate=(-5, 5), p=0.4),
            A.ElasticTransform(alpha=1, sigma=50, p=0.2),
        ]
    return A.Compose(
        transforms,
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3),
    )


def _read_yolo_labels(label_path: Path) -> tuple:
    """读取 YOLO 格式标签，返回 (classes_list, bboxes_list)."""
    classes, bboxes = [], []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                classes.append(int(float(parts[0])))
                bboxes.append([float(x) for x in parts[1:5]])
    return classes, bboxes


def _write_yolo_labels(label_path: Path, classes, bboxes):
    """写入 YOLO 格式标签."""
    with open(label_path, "w") as f:
        for cls_id, bbox in zip(classes, bboxes):
            f.write(f"{cls_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")


def oversample_defects(
    label_dir: str,
    target_count: int = 200,
    oversample_factor: int = 4,
    use_augmentation: bool = True,
):
    """
    对包含小样本类别的图片进行过采样.

    Args:
        label_dir: 标签目录路径（如 data/labels/train）
        target_count: 目标样本数（暂未使用，保留兼容）
        oversample_factor: 每张图片额外生成的变体数量
        use_augmentation: 是否使用 albumentations 增强（否则仅复制）
    """
    label_path = Path(label_dir)
    image_dir = label_path.parent.parent / "images" / label_path.name

    # 收集包含小样本类别的图片（排除已生成的 _aug / _copy 文件，防止重复膨胀）
    matched_images = []
    for lbl_file in sorted(label_path.glob("*.txt")):
        stem = lbl_file.stem
        if "_aug" in stem or "_copy" in stem:
            continue
        classes, _ = _read_yolo_labels(lbl_file)
        if any(cls_id in LOW_SAMPLE_IDS for cls_id in classes):
            matched_images.append(stem)

    logger.info(f"Found {len(matched_images)} images containing low-sample classes {LOW_SAMPLE_IDS}.")

    if len(matched_images) == 0:
        logger.info("No new images to oversample — all done.")
        return

    generated = 0
    for stem in tqdm(matched_images, desc="Oversampling", unit="img"):
        lbl_path = label_path / f"{stem}.txt"
        # 定位图片文件
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            candidate = image_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            logger.warning(f"Image not found for {stem}, skipping.")
            continue

        orig_classes, orig_bboxes = _read_yolo_labels(lbl_path)

        has_box = 9 in orig_classes  # box 类对几何变形敏感

        # 类别倍数: damage ×3, wrinkle ×2, 取最大值
        max_mult = 1
        for cid in orig_classes:
            if cid in CLASS_MULTIPLIER:
                max_mult = max(max_mult, CLASS_MULTIPLIER[cid])
        actual_factor = oversample_factor * max_mult

        if use_augmentation:
            augment = _get_augmentation_pipeline(box_safe=has_box)
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning(f"Cannot read image {img_path}, skipping.")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]

            # 转为 corner 格式 clamp 后再转回 YOLO，保证角点全在 [0,1]
            safe_bboxes = _clamp_bboxes_yolo(orig_bboxes)

            for i in range(actual_factor):
                transformed = augment(image=image, bboxes=[b.copy() for b in safe_bboxes], class_labels=orig_classes.copy())
                aug_img = transformed["image"]
                # 增强后再次用 corner-clamp，因为 ElasticTransform 可能推出边界
                aug_bboxes = _clamp_bboxes_yolo(transformed["bboxes"])
                aug_classes = transformed["class_labels"]

                new_stem = f"{stem}_aug{i}"
                # 保存增强后的图片
                aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(image_dir / f"{new_stem}.jpg"), aug_img_bgr)
                # 保存增强后的标签（bbox 可能因翻转/旋转而有微小变化）
                _write_yolo_labels(label_path / f"{new_stem}.txt", aug_classes, aug_bboxes)
                generated += 1
        else:
            for i in range(actual_factor):
                new_stem = f"{stem}_copy{i}"
                shutil.copy(lbl_path, label_path / f"{new_stem}.txt")
                shutil.copy(img_path, image_dir / f"{new_stem}{img_path.suffix}")
                generated += 1

    logger.info(f"Oversampling completed — generated {generated} new samples.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Oversample low-sample classes (damage, wrinkle, box)")
    parser.add_argument("--label_dir", default="./data/labels/train", help="Path to training labels directory")
    parser.add_argument("--factor", type=int, default=4, help="Number of augmented variants per image")
    args = parser.parse_args()
    oversample_defects(label_dir=args.label_dir, oversample_factor=args.factor)
