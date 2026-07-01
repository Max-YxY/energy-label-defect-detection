#!/usr/bin/env python3
"""
Comprehensive model evaluation script.
Tests ALL .pt models in the project directory on the validation set.
Handles different architectures (yolov8, yolo11, yolo26, etc.) automatically.
Ranks by mAP50-95 (primary) and mAP50.
"""
import os
import sys
import time
import json
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── GPU check ──
import subprocess as _sp
try:
    _r = _sp.run([sys.executable, "-c",
        "import torch; print(torch.cuda.is_available())"],
        capture_output=True, text=True, timeout=15)
    _gpu_ok = _r.stdout.strip() == "True"
except Exception:
    _gpu_ok = False

print(f"GPU available: {_gpu_ok}")

import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(r"E:\energy_label_defect_detection\P")
DATA_YAML = str(PROJECT_ROOT / "data" / "dataset.yaml")
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "model_eval"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Exclude these models from testing (wrong task or known incompatible)
EXCLUDE_PATTERNS = [
    "box_detector.pt",        # Single-class box detector, not 10-class
    "yolov8n-seg.pt",         # Segmentation model, needs task='segment'
    "last.pt",                # Skip intermediate checkpoints, only test best.pt
]

def get_model_info(model_path: str) -> Dict:
    """Extract model metadata without running inference."""
    info = {
        "path": model_path,
        "size_mb": os.path.getsize(model_path) / (1024 * 1024),
        "architecture": "unknown",
        "task": "unknown",
        "nc": None,
        "loadable": False,
        "error": None,
    }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = YOLO(model_path, verbose=False)

        # Determine task
        if hasattr(model, "task") and model.task:
            info["task"] = model.task
        elif hasattr(model.model, "task"):
            info["task"] = model.model.task
        else:
            # Infer from model attributes
            if hasattr(model.model, "names"):
                info["task"] = "detect"  # default

        # Get class count
        if hasattr(model.model, "model") and hasattr(model.model.model, "nc"):
            info["nc"] = model.model.model.nc
        elif hasattr(model.model, "nc"):
            info["nc"] = model.model.nc

        # Get architecture name
        if hasattr(model.model, "yaml"):
            yaml_info = model.model.yaml
            if isinstance(yaml_info, dict):
                info["architecture"] = yaml_info.get("yaml_file", "unknown")
            elif isinstance(yaml_info, str):
                info["architecture"] = yaml_info

        # Try to get model name from file
        model_name = Path(model_path).stem
        if "yolov8" in model_name.lower():
            info["architecture"] = "yolov8"
        elif "yolo11" in model_name.lower():
            info["architecture"] = "yolo11"
        elif "yolo26" in model_name.lower():
            info["architecture"] = "yolo26"

        info["loadable"] = True

        # Clean up
        del model
        if _gpu_ok:
            torch.cuda.empty_cache()

    except Exception as e:
        info["error"] = str(e)[:200]

    return info


def evaluate_model(model_path: str, imgsz: int = 640) -> Optional[Dict]:
    """Run full evaluation on one model. Returns metrics dict or None on failure."""
    print(f"\n{'='*70}")
    print(f"Evaluating: {model_path}")
    print(f"{'='*70}")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = YOLO(model_path, verbose=False)

        # Determine appropriate task and parameters
        task = getattr(model, "task", None)
        if task == "segment":
            print("  ⚠ Segmentation model, not compatible with detection eval. Skipping.")
            del model
            return None
        if task == "classify":
            print("  ⚠ Classification model, not compatible with detection eval. Skipping.")
            del model
            return None
        if task == "pose":
            print("  ⚠ Pose model, not compatible with detection eval. Skipping.")
            del model
            return None

        # Check number of classes matches dataset
        model_nc = None
        if hasattr(model.model, "model") and hasattr(model.model.model, "nc"):
            model_nc = model.model.model.nc
        elif hasattr(model.model, "nc"):
            model_nc = model.model.nc

        if model_nc is not None and model_nc != 10:
            print(f"  ⚠ Model has {model_nc} classes, dataset has 10. May produce incomplete results.")

        t0 = time.perf_counter()
        metrics = model.val(
            data=DATA_YAML,
            imgsz=imgsz,
            conf=0.25,
            iou=0.5,
            split="val",
            verbose=False,
            workers=0,  # Windows safety
        )
        elapsed = time.perf_counter() - t0

        result = {
            "path": model_path,
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "eval_time_sec": round(elapsed, 1),
            "model_nc": model_nc,
        }

        # Per-class AP50
        per_class = {}
        names = {0:"level_1",1:"level_2",2:"level_3",3:"level_4",4:"level_5",
                 5:"stain",6:"damage",7:"wrinkle",8:"label",9:"box"}
        if hasattr(metrics.box, "ap_class_index") and metrics.box.ap_class_index is not None:
            ap_indices = metrics.box.ap_class_index
            if hasattr(ap_indices, "cpu"):
                ap_indices = ap_indices.cpu().numpy()

            if hasattr(metrics.box, "ap50") and metrics.box.ap50 is not None:
                ap_arr = metrics.box.ap50
            else:
                ap_arr = metrics.box.ap
            if hasattr(ap_arr, "cpu"):
                ap_arr = ap_arr.cpu().numpy()

            for idx, ap_val in zip(ap_indices, ap_arr):
                name = names.get(int(idx), f"class_{idx}")
                per_class[name] = round(float(ap_val), 4)

        result["per_class_AP50"] = per_class

        # Add F1 score (harmonic mean of precision and recall)
        p, r = result["precision"], result["recall"]
        result["F1"] = round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0

        # Print summary
        print(f"  mAP50:    {result['mAP50']:.4f}")
        print(f"  mAP50-95: {result['mAP50_95']:.4f}")
        print(f"  Precision:{result['precision']:.4f}")
        print(f"  Recall:   {result['recall']:.4f}")
        print(f"  F1:       {result['F1']:.4f}")
        print(f"  Time:     {elapsed:.1f}s")

        del model
        if _gpu_ok:
            torch.cuda.empty_cache()

        return result

    except Exception as e:
        print(f"  ✗ FAILED: {type(e).__name__}: {str(e)[:300]}")
        del model
        if _gpu_ok:
            torch.cuda.empty_cache()
        return None


def should_exclude(path: str) -> bool:
    """Check if model should be excluded."""
    fname = os.path.basename(path)
    for pattern in EXCLUDE_PATTERNS:
        if pattern.lower() in fname.lower():
            return True
    return False


def main():
    print("=" * 70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print(f"Dataset: {DATA_YAML}")
    print(f"GPU: {_gpu_ok}")
    print("=" * 70)

    # ── Phase 1: Discover all models ──
    print("\n📁 Phase 1: Discovering all .pt models...")
    all_models = []
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Skip __pycache__ and non-model dirs
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", "node_modules")]
        for f in files:
            if f.endswith(".pt") and not should_exclude(f):
                full_path = os.path.join(root, f)
                all_models.append(full_path)

    all_models = sorted(set(all_models))
    print(f"Found {len(all_models)} candidate .pt models (after exclusions)")

    # ── Phase 2: Quick model info ──
    print("\n📋 Phase 2: Extracting model metadata...")
    model_infos = []
    for mp in all_models:
        info = get_model_info(mp)
        rel_path = os.path.relpath(mp, PROJECT_ROOT)
        status = "✅" if info["loadable"] else "❌"
        print(f"  {status} {rel_path}")
        print(f"     Size: {info['size_mb']:.1f}MB | Architecture: {info['architecture']} | "
              f"Task: {info['task']} | Classes: {info['nc']}")
        if info["error"]:
            print(f"     Error: {info['error'][:120]}")
        model_infos.append(info)

    # ── Phase 3: Full evaluation ──
    print("\n🚀 Phase 3: Running full evaluation...")
    results = []
    errors = []

    for info in model_infos:
        mp = info["path"]
        rel_path = os.path.relpath(mp, PROJECT_ROOT)

        # Skip non-detection models
        if info["task"] not in ("detect", "unknown", None, ""):
            print(f"\n  ⏭ Skipping {rel_path} (task={info['task']}, not detection)")
            errors.append({"path": rel_path, "reason": f"Non-detect task: {info['task']}"})
            continue

        # Skip unloadable models
        if not info["loadable"]:
            print(f"\n  ⏭ Skipping {rel_path} (not loadable: {info.get('error', 'unknown')[:100]})")
            errors.append({"path": rel_path, "reason": f"Not loadable: {info.get('error', 'unknown')[:100]}"})
            continue

        result = evaluate_model(mp)
        if result:
            result["rel_path"] = rel_path
            result["size_mb"] = info["size_mb"]
            result["architecture"] = info["architecture"]
            results.append(result)
        else:
            errors.append({"path": rel_path, "reason": "Evaluation failed (see above)"})

    # ── Phase 4: Rank and report ──
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS - RANKED BY mAP50-95")
    print("=" * 70)

    results.sort(key=lambda r: r["mAP50_95"], reverse=True)

    # Print leaderboard
    print(f"\n{'Rank':<5} {'Model':<55} {'mAP50':<9} {'mAP50-95':<10} {'Prec':<9} {'Recall':<9} {'F1':<9}")
    print("-" * 110)
    for rank, r in enumerate(results, 1):
        crown = "👑" if rank == 1 else "  "
        print(f"{crown}{rank:<3} {r['rel_path']:<55} {r['mAP50']:.4f}   {r['mAP50_95']:.4f}    "
              f"{r['precision']:.4f}   {r['recall']:.4f}   {r['F1']:.4f}")

    # Best model details
    if results:
        best = results[0]
        print(f"\n{'='*70}")
        print(f"🏆 BEST MODEL: {best['rel_path']}")
        print(f"{'='*70}")
        print(f"  Architecture: {best['architecture']}")
        print(f"  Size:         {best['size_mb']:.1f} MB")
        print(f"  mAP50:        {best['mAP50']:.4f}")
        print(f"  mAP50-95:     {best['mAP50_95']:.4f}")
        print(f"  Precision:    {best['precision']:.4f}")
        print(f"  Recall:       {best['recall']:.4f}")
        print(f"  F1 Score:     {best['F1']:.4f}")

        if best.get("per_class_AP50"):
            print(f"\n  Per-class AP50:")
            for cls_name, ap in sorted(best["per_class_AP50"].items()):
                print(f"    {cls_name:<12} {ap:.4f}")

    # Skipped/errors
    if errors:
        print(f"\n⚠ Skipped/Failed ({len(errors)}):")
        for e in errors:
            print(f"  • {e['path']}: {e['reason']}")

    # ── Save results ──
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"eval_results_{timestamp}.json"
    full_report = {
        "eval_time": timestamp,
        "dataset": DATA_YAML,
        "gpu_available": _gpu_ok,
        "total_models_tested": len(results),
        "total_skipped": len(errors),
        "ranked_results": results,
        "errors": errors,
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Full report saved to: {output_file}")

    # Also save a quick summary
    summary_file = OUTPUT_DIR / "latest_leaderboard.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)

    return results, best if results else None


if __name__ == "__main__":
    main()
