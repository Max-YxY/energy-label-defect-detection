#!/usr/bin/env python3
"""
"检出就算" 评测脚本 — 匹配项目的实际检测算法逻辑。
对每张图，有 GT 标注的类别，模型检出一个就算成功。
不关心 bbox 精度，只关心"有没有检出"。
"""
import os, sys, json, time, warnings
from pathlib import Path
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ultralytics import YOLO

PROJECT = Path(r"E:\energy_label_defect_detection\P")
VAL_IMG = PROJECT / "data/images/val"
VAL_LBL = PROJECT / "data/labels/val"
OUTPUT = PROJECT / "outputs" / "detect_or_not_eval"
OUTPUT.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {0:"level_1",1:"level_2",2:"level_3",3:"level_4",4:"level_5",
               5:"stain",6:"damage",7:"wrinkle",8:"label",9:"box"}

# ── Step 1: 构建 GT 索引 ──
print("Building GT index...")
gt_index = defaultdict(set)       # class_id -> set of image stems
gt_all_images = set()

for lbl_file in sorted(VAL_LBL.glob("*.txt")):
    stem = lbl_file.stem
    gt_all_images.add(stem)
    with open(lbl_file) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                cls_id = int(float(parts[0]))
                gt_index[cls_id].add(stem)

# 找到图片路径
def find_img(stem):
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        p = VAL_IMG / f"{stem}{ext}"
        if p.exists():
            return str(p)
    return None

# ── Step 2: 发现所有模型 ──
EXCLUDE = ["box_detector.pt", "yolov8n-seg.pt", "last.pt"]
model_paths = []
for root, dirs, files in os.walk(PROJECT):
    dirs[:] = [d for d in dirs if d not in ("__pycache__",)]
    for f in files:
        if f.endswith(".pt") and not any(e in f.lower() for e in EXCLUDE):
            model_paths.append(os.path.join(root, f))
model_paths = sorted(set(model_paths))
print(f"Found {len(model_paths)} models to test")

# ── Step 3: 逐模型评测 ──
results = []
CONF = 0.25   # 低阈值，尽量检出

for mp in model_paths:
    rel = os.path.relpath(mp, PROJECT)
    print(f"\n{'='*60}")
    print(f"Testing: {rel}")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = YOLO(mp, verbose=False)
    except Exception as e:
        print(f"  SKIP: cannot load ({e})")
        continue

    # 获取模型类别数
    try:
        nc = model.model.model.nc if hasattr(model.model.model, 'nc') else model.model.nc
    except:
        nc = None
    if nc and nc != 10:
        print(f"  SKIP: nc={nc} (not 10-class)")
        continue

    # 逐图推理
    detected = defaultdict(set)  # class_id -> set of stems where detected
    total_images = 0
    errors = 0

    for stem in sorted(gt_all_images):
        img_path = find_img(stem)
        if not img_path:
            continue
        total_images += 1

        try:
            res = model(img_path, conf=CONF, iou=0.45, imgsz=640, verbose=False)
            if res[0].boxes is not None:
                for cls_id in res[0].boxes.cls.int().tolist():
                    detected[int(cls_id)].add(stem)
        except:
            errors += 1

    # 计算每类"检出就算"召回率
    per_class_recall = {}
    for cls_id in range(10):
        gt_set = gt_index.get(cls_id, set())
        det_set = detected.get(cls_id, set())
        if len(gt_set) > 0:
            per_class_recall[CLASS_NAMES[cls_id]] = len(det_set & gt_set) / len(gt_set)
        else:
            per_class_recall[CLASS_NAMES[cls_id]] = None  # 无GT

    # 汇总指标
    valid_recalls = [v for v in per_class_recall.values() if v is not None]
    mean_recall_all = sum(valid_recalls) / len(valid_recalls) if valid_recalls else 0.0
    mean_recall_no_box = sum(v for k, v in per_class_recall.items()
                            if v is not None and k != "box") / max(1, sum(1 for k, v in per_class_recall.items()
                            if v is not None and k != "box"))

    r = {
        "model": rel,
        "size_mb": os.path.getsize(mp) / (1024*1024),
        "total_images": total_images,
        "errors": errors,
        "mean_recall_all": round(mean_recall_all, 4),
        "mean_recall_no_box": round(mean_recall_no_box, 4),
        "per_class_recall": {k: round(v, 4) if v is not None else None
                            for k, v in per_class_recall.items()},
    }
    results.append(r)

    print(f"  mean_recall(all 10):  {mean_recall_all:.4f}")
    print(f"  mean_recall(no box):  {mean_recall_no_box:.4f}")
    for cls_name, recall in per_class_recall.items():
        if recall is not None:
            bar = "█" * int(recall * 20) + "░" * (20 - int(recall * 20))
            print(f"    {cls_name:<10} {recall:.4f} {bar}")

    del model

# ── Step 4: 排名 ──
results.sort(key=lambda r: r["mean_recall_no_box"], reverse=True)

print(f"\n{'='*80}")
print("FINAL RANKING — '检出就算' Recall (排除 box)")
print(f"{'='*80}")
print(f"{'Rank':<5} {'Model':<52} {'9cls Recall':<12} {'All Recall':<11} {'damage':<8} {'wrinkle':<8}")
print("-" * 90)
for i, r in enumerate(results):
    tag = ">>" if i == 1 else "  "
    dam = r["per_class_recall"].get("damage", 0) or 0
    wrk = r["per_class_recall"].get("wrinkle", 0) or 0
    print(f"{tag}{i:<3} {r['model']:<52} {r['mean_recall_no_box']:<12.4f} "
          f"{r['mean_recall_all']:<11.4f} {dam:<8.4f} {wrk:<8.4f}")

# Best model
if results:
    best = results[0]
    print(f"\n🏆 BEST: {best['model']}")
    print(f"   9-class mean recall (no box): {best['mean_recall_no_box']:.4f}")
    print(f"   Per-class recall:")
    for cls_name, recall in sorted(best["per_class_recall"].items()):
        if recall is not None:
            print(f"     {cls_name:<10} {recall:.4f}")

# Save
report = {
    "eval_time": time.strftime("%Y%m%d_%H%M%S"),
    "method": "detect-or-not — 检出就算",
    "conf_threshold": CONF,
    "val_images": len(gt_all_images),
    "ranked_results": results,
}
ts = time.strftime("%Y%m%d_%H%M%S")
with open(OUTPUT / f"detect_or_not_{ts}.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
with open(OUTPUT / "latest_detect_or_not.json", "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f"\nSaved to: {OUTPUT}")
