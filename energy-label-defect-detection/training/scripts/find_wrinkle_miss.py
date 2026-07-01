"""
定位 wrinkle 漏检图 — 在 val 集中找出 YOLO 未检出 wrinkle 的图片。
"""
import os
import sys
from pathlib import Path
import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path.cwd()))

BASE = Path(r"E:\energy_label_defect_detection\P")
MODEL_PATH = str(BASE / "models/best.pt")
VAL_IMG_DIR = BASE / "data/images/val"
VAL_LBL_DIR = BASE / "data/labels/val"
WRINKLE_ID = 7
CONF = 0.1
IOU = 0.45


def main():
    model = YOLO(MODEL_PATH)

    # 找出所有包含 wrinkle(id=7) 标注的 val 图片
    wrinkle_images = []
    for lbl_file in sorted(VAL_LBL_DIR.glob("*.txt")):
        with open(lbl_file) as f:
            for line in f:
                parts = line.strip().split()
                if parts and int(float(parts[0])) == WRINKLE_ID:
                    wrinkle_images.append(lbl_file.stem)
                    break

    print(f"Val 集中有 wrinkle 标注的图片: {len(wrinkle_images)} 张")
    print("=" * 60)

    missed = []
    detected = []

    for stem in wrinkle_images:
        # 定位图片
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            p = VAL_IMG_DIR / f"{stem}{ext}"
            if p.exists():
                img_path = p
                break
        if img_path is None:
            print(f"  ⚠️  找不到图片: {stem}")
            continue

        results = model(str(img_path), conf=CONF, iou=IOU, imgsz=640, verbose=False)

        wrinkle_found = False
        if results[0].boxes is not None:
            for cls_id in results[0].boxes.cls.int().tolist():
                if cls_id == WRINKLE_ID:
                    wrinkle_found = True
                    break

        if wrinkle_found:
            detected.append(stem)
        else:
            missed.append(stem)
            # 读取标注信息
            with open(VAL_LBL_DIR / f"{stem}.txt") as f:
                lbl_lines = f.readlines()
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            print(f"\n🔴 漏检: {stem}")
            print(f"   图片尺寸: {w}×{h}")
            print(f"   标注行数: {len(lbl_lines)}")
            for line in lbl_lines:
                parts = line.strip().split()
                if parts:
                    cid = int(float(parts[0]))
                    xc, yc, bw, bh = [float(x) for x in parts[1:5]]
                    if cid == WRINKLE_ID:
                        print(f"   wrinkle bbox: xc={xc:.3f} yc={yc:.3f} w={bw:.3f} h={bh:.3f} → {int(xc*w):.0f},{int(yc*h):.0f} {int(bw*w):.0f}×{int(bh*h):.0f}px")
            # 保存漏检图副本到当前目录以便查看
            save_path = str(BASE / f"missed_wrinkle_{stem}.jpg")
            cv2.imwrite(save_path, img)
            print(f"   已保存副本: {save_path}")

    print(f"\n{'=' * 60}")
    print(f"检出: {len(detected)}/{len(wrinkle_images)}")
    print(f"漏检: {len(missed)}/{len(wrinkle_images)}")
    if missed:
        print(f"漏检图: {missed}")
        print(f"\n漏检图已保存至项目根目录: {BASE}")


if __name__ == "__main__":
    main()
