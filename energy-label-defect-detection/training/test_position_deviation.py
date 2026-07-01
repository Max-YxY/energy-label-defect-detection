"""
位置偏差专项测试 — 对指定目录的每张图跑检测，统计偏差检出率。
用法: python test_position_deviation.py
"""
import sys
sys.path.insert(0, r"E:\energy_label_defect_detection\P")

import time
import json
from pathlib import Path
import cv2

from inference.detector import EnergyLabelDetector

# ── 配置 ──
PROJECT = Path(r"E:\energy_label_defect_detection\P")
DEVIATION_DIR = Path(r"C:\Users\ASUS\Desktop\位置偏差good")
OUTPUT_DIR = PROJECT / "outputs" / "test_results"

print("=" * 60)
print("位置偏差检测测试")
print(f"数据目录: {DEVIATION_DIR}")
print("=" * 60)

# 加载检测器
print("\n加载模型...")
detector = EnergyLabelDetector(str(PROJECT / "config.yaml"))

# 扫描图片
images = sorted(
    [p for p in DEVIATION_DIR.iterdir()
     if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
)
print(f"共找到 {len(images)} 张图片")

# 逐张检测
results = {
    "test_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    "model": "models/best.pt + box_detector.pt + CV fallback",
    "config": {
        "tolerance_x": 0.18,
        "tolerance_y": 0.155,
        "edge_margin": 0.01,
        "confidence_threshold": 0.3,
    },
    "total_images": len(images),
    "details": []
}

stats = {
    "total": 0,
    "has_label": 0,
    "has_box": 0,
    "has_both": 0,        # 同时有 label 和 box
    "position_deviation": 0,
    "double_stage_used": 0,  # 走了两阶段回退的
    "errors": 0,
}

for i, img_path in enumerate(images):
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            stats["errors"] += 1
            continue

        result = detector.detect(img)
        stats["total"] += 1

        has_label = any(b["class_id"] == 8 for b in result["boxes"])
        has_box = any(b["class_id"] == 9 for b in result["boxes"])
        has_both = has_label and has_box
        is_dev = result["position_deviation"]

        if has_label:
            stats["has_label"] += 1
        if has_box:
            stats["has_box"] += 1
        if has_both:
            stats["has_both"] += 1
        if is_dev:
            stats["position_deviation"] += 1

        # 检查是否用了两阶段回退（box出现在label区域内且conf<0.5）
        box_boxes = [b for b in result["boxes"] if b["class_id"] == 9]
        if box_boxes and any(b["confidence"] < 0.5 for b in box_boxes):
            stats["double_stage_used"] += 1

        detail = {
            "file": img_path.name,
            "has_label": has_label,
            "has_box": has_box,
            "position_deviation": is_dev,
            "offset_x": result["offset_x"],
            "offset_y": result["offset_y"],
            "energy_level": result["energy_level"],
            "defects": [d["defect_type"] for d in result.get("defects", [])],
            "inference_ms": round(result["inference_time_ms"], 2),
        }
        results["details"].append(detail)

        # 进度
        status = "⚠️ DEVIATED" if is_dev else "✅ OK"
        box_status = "📦" if has_box else "❌NOBOX"
        print(f"  [{i+1:3d}/{len(images)}] {img_path.name:45s} {box_status} {status}  "
              f"dx={result['offset_x']:+.4f} dy={result['offset_y']:+.4f}")

    except Exception as e:
        stats["errors"] += 1
        print(f"  [{i+1:3d}/{len(images)}] {img_path.name:45s} ❌ ERROR: {e}")

# ── 汇总 ──
print("\n" + "=" * 60)
print("统计汇总")
print("=" * 60)

n = stats["total"]
print(f"  总图片数:          {n}")
print(f"  检测到 label:      {stats['has_label']}  ({stats['has_label']/max(n,1)*100:.1f}%)")
print(f"  检测到 box:        {stats['has_box']}  ({stats['has_box']/max(n,1)*100:.1f}%)")
print(f"  同时有 label+box:  {stats['has_both']}  ({stats['has_both']/max(n,1)*100:.1f}%)")
print(f"  两阶段回退触发:    {stats['double_stage_used']}")

print(f"\n  🔴 位置偏差检出:   {stats['position_deviation']} / {stats['has_both']} "
      f"({stats['position_deviation']/max(stats['has_both'],1)*100:.1f}%)")

no_box = [d for d in results["details"] if not d["has_box"]]
ok_but_dev = [d for d in results["details"] if not d["has_box"] and d["position_deviation"]]
print(f"  无 box 的图:       {len(no_box)}")
if ok_but_dev:
    print(f"   其中仍然判偏:     {len(ok_but_dev)} (无box却能判偏?)")

# ── 假偏分析: 有 box 但未判偏 ──
not_dev = [d for d in results["details"] if d["has_box"] and not d["position_deviation"]]
print(f"\n  有 box 但未判偏:   {len(not_dev)}")
if not_dev:
    print(f"  (可能漏检)")

# ── 保存结果 ──
results["stats"] = stats
timestamp = time.strftime("%Y%m%d_%H%M%S")
output_file = OUTPUT_DIR / f"position_deviation_test_{timestamp}.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\n详细结果已保存: {output_file}")

# 保存最新
with open(OUTPUT_DIR / "latest_position_deviation.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("=" * 60)
