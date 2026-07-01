"""Test runner — runs evaluation, benchmark, stability on the optimal model."""
import sys
sys.path.insert(0, r"E:\energy_label_defect_detection\P")

import time
import json
from pathlib import Path

PROJECT = Path(r"E:\energy_label_defect_detection\P")
OUTPUT = PROJECT / "outputs" / "test_results"
OUTPUT.mkdir(parents=True, exist_ok=True)


def run_all_tests():
    results = {
        "test_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "models/best.pt (train6, yolov8n)",
        "tests": {}
    }

    # ── Test 1: Evaluation ──
    print("=" * 60)
    print("TEST 1: Model Evaluation (val set, 285 images)")
    print("=" * 60)
    from scripts.evaluate import evaluate
    metrics = evaluate(
        str(PROJECT / "models/best.pt"),
        str(PROJECT / "data/dataset.yaml"),
        imgsz=640,
        workers=0  # Windows: avoid multiprocessing
    )
    results["tests"]["evaluation"] = {
        "mAP50": f"{metrics.box.map50:.4f}",
        "mAP50-95": f"{metrics.box.map:.4f}",
        "precision": f"{metrics.box.mp:.4f}",
        "recall": f"{metrics.box.mr:.4f}"
    }
    if hasattr(metrics.box, "ap_class_index") and metrics.box.ap_class_index is not None:
        per_class = {}
        ap_arr = (metrics.box.ap50 if hasattr(metrics.box, "ap50") and metrics.box.ap50 is not None 
                  else metrics.box.ap)
        if hasattr(ap_arr, "cpu"):
            ap_arr = ap_arr.cpu().numpy()
        indices = metrics.box.ap_class_index
        if hasattr(indices, "cpu"):
            indices = indices.cpu().numpy()
        names = {0:"level_1",1:"level_2",2:"level_3",3:"level_4",4:"level_5",
                 5:"stain",6:"damage",7:"wrinkle",8:"label",9:"box"}
        for idx, ap in zip(indices, ap_arr):
            name = names.get(int(idx), f"class_{idx}")
            per_class[name] = f"{float(ap):.4f}"
        results["tests"]["evaluation"]["per_class_AP50"] = per_class

    # ── Test 2: Benchmark ──
    print("\n" + "=" * 60)
    print("TEST 2: Inference Benchmark (50 frames)")
    print("=" * 60)
    from scripts.benchmark import benchmark
    bench_result = benchmark(str(PROJECT / "config.yaml"), num_frames=50)
    results["tests"]["benchmark"] = {
        "avg_latency_ms": f"{bench_result['avg_ms']:.2f}",
        "fps": f"{bench_result['fps']:.1f}"
    }

    # ── Test 3: Quick Stability ──
    print("\n" + "=" * 60)
    print("TEST 3: Quick Stability (1 minute)")
    print("=" * 60)
    from scripts.stability_test import StabilityTester
    tester = StabilityTester(str(PROJECT / "config.yaml"), duration_minutes=1)
    tester.run()
    results["tests"]["stability"] = {
        "duration_minutes": 1,
        "errors": tester.error_count
    }

    # ── Save results ──
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT / f"test_results_{timestamp}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_file}")

    with open(OUTPUT / "latest_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Also saved as: latest_results.json")

    return results

if __name__ == "__main__":
    run_all_tests()
