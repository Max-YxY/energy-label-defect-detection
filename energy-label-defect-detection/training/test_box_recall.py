"""
Box 检出率测试 — 在 val 集有 box GT 的 29 张图上，使用三阶段管线评估实际 box 检出率。
"""
import sys
sys.path.insert(0, r"E:\energy_label_defect_detection\P")
from pathlib import Path
import cv2

from inference.detector import EnergyLabelDetector

PROJECT = Path(r"E:\energy_label_defect_detection\P")
VAL_IMG = PROJECT / "data/images/val"
VAL_LBL = PROJECT / "data/labels/val"

print("=" * 60)
print("Box 检出率测试 — 三阶段管线 (YOLO→Box Detector→CV Fallback)")
print(f"Val 目录: {VAL_IMG}")
print("=" * 60)

# 找出有 box (class_id=9) GT 标注的 val 图片
box_images = []
for lbl_file in sorted(VAL_LBL.glob("*.txt")):
    with open(lbl_file) as f:
        for line in f:
            parts = line.strip().split()
            if parts and int(float(parts[0])) == 9:  # box
                box_images.append(lbl_file.stem)
                break

print(f"\nVal 集中有 box GT 的图片: {len(box_images)} 张")

# 加载检测器
detector = EnergyLabelDetector(str(PROJECT / "config.yaml"))

box_found = 0
box_missed = []
details = []

for stem in box_images:
    # 找图片
    img_path = None
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        p = VAL_IMG / f"{stem}{ext}"
        if p.exists():
            img_path = p
            break
    if img_path is None:
        print(f"  ⚠️ 找不到图片: {stem}")
        continue

    img = cv2.imread(str(img_path))
    result = detector.detect(img)

    has_box = any(b["class_id"] == 9 for b in result["boxes"])
    box_boxes = [b for b in result["boxes"] if b["class_id"] == 9]

    if has_box:
        box_found += 1
        confs = [b["confidence"] for b in box_boxes]
        source = "YOLO" if max(confs) > 0.3 else "fallback"
        print(f"  ✅ {stem}: box found (conf={max(confs):.4f}, via {source})")
    else:
        box_missed.append(stem)
        print(f"  ❌ {stem}: box NOT found!")

    details.append({
        "file": stem,
        "box_found": has_box,
        "box_confs": [b["confidence"] for b in box_boxes],
    })

print(f"\n{'=' * 60}")
print(f"Box 检出率: {box_found}/{len(box_images)} = {box_found/len(box_images)*100:.1f}%")
if box_missed:
    print(f"漏检: {box_missed}")
print(f"{'=' * 60}")
