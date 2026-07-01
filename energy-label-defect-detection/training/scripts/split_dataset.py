"""
用法：python scripts/split_dataset.py
功能：从 images/train + labels/train 中按比例随机分出 val 集
默认 8:2 分割（20% → val）
"""
import os
import shutil
import random
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # 项目根目录 P/

SRC_IMAGES = BASE_DIR / "data" / "images" / "train"
SRC_LABELS = BASE_DIR / "data" / "labels" / "train"
DST_IMAGES = BASE_DIR / "data" / "images" / "val"
DST_LABELS = BASE_DIR / "data" / "labels" / "val"

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def main():
    parser = argparse.ArgumentParser(description="数据集分割 train→val")
    parser.add_argument("--ratio", type=float, default=0.2,
                        help="验证集占比，默认 0.2（即 8:2 分割）")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子，保证结果可复现")
    args = parser.parse_args()

    random.seed(args.seed)

    # 收集所有图片
    all_images = sorted([
        p for p in SRC_IMAGES.iterdir()
        if p.suffix.lower() in VALID_EXTS
    ])
    if not all_images:
        print(f"❌ {SRC_IMAGES} 下无图片！")
        return

    print(f"📁 共发现 {len(all_images)} 张图片")

    # 随机抽取 val 集
    n_val = max(1, int(len(all_images) * args.ratio))
    val_set = set(random.sample(all_images, n_val))

    # 创建目标目录
    DST_IMAGES.mkdir(parents=True, exist_ok=True)
    DST_LABELS.mkdir(parents=True, exist_ok=True)

    moved_img, moved_lbl, no_lbl = 0, 0, 0

    for img in all_images:
        stem = img.stem
        lbl = SRC_LABELS / f"{stem}.txt"

        if img in val_set:
            # 移动图片
            shutil.move(str(img), str(DST_IMAGES / img.name))
            moved_img += 1
            # 移动同名标注
            if lbl.exists():
                shutil.move(str(lbl), str(DST_LABELS / f"{stem}.txt"))
                moved_lbl += 1
            else:
                no_lbl += 1
                print(f"⚠️  {img.name} 缺少标注文件")

    print(f"✅ 分割完成！（比例 {1 - args.ratio:.0%}:{args.ratio:.0%}，种子 {args.seed}）")
    print(f"   train → {len(all_images) - moved_img} 张图片")
    print(f"   val   → {moved_img} 张图片 (+ {moved_lbl} 标注)")
    if no_lbl:
        print(f"   ⚠️  {no_lbl} 张图片缺少标注")


if __name__ == "__main__":
    main()