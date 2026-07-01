#!/usr/bin/env python3
"""主入口程序."""
import os as _os
_os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys as _sys
# ---- GPU pre-check: must happen BEFORE any torch import ----
import subprocess, json
try:
    _r = subprocess.run([_sys.executable, "-c", "import torch; print(json.dumps({'ok':torch.cuda.is_available(),'ver':torch.__version__,'dev':torch.cuda.device_count()}))"], capture_output=True, text=True, timeout=15)
    _info = json.loads(_r.stdout.strip())
    if not _info["ok"]:
        _sys.exit(f"FATAL: PyTorch {_info['ver']} has NO CUDA. Reinstall: pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall")
except Exception:
    _os.environ["CUDA_VISIBLE_DEVICES"] = ""
# ----------------------------------------------------------------
import argparse
from pathlib import Path
_sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.logger import setup_logger, get_logger

logger = get_logger(__name__)


def mode_train(args):
    from training import LabelDefectTrainer
    trainer = LabelDefectTrainer(args.config)
    trainer.run(resume=args.resume)


def mode_detect(args):
    import cv2
    from inference.detector import EnergyLabelDetector
    from utils.database import DetectionDatabase

    detector = EnergyLabelDetector(args.config)

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            logger.error(f"Cannot read image: {args.image}")
            return
        result = detector.detect(img)
        annotated = detector.draw_results(img, result)
        output_path = args.output or "output.jpg"
        cv2.imwrite(output_path, annotated)
        logger.info(f"Result saved to {output_path}")
        logger.info(f"Energy Level: {result['energy_level']}")
        logger.info(f"Defects: {result['defects']}")
        logger.info(f"Position Deviation: {result['position_deviation']}")
        return

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        logger.error("Cannot open camera.")
        return

    db = DetectionDatabase()
    logger.info("Press 'q' or ESC to quit.")
    counter = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result = detector.detect(frame)
            annotated = detector.draw_results(frame, result)
            import time
            product_id = f"PC-{time.strftime('%Y%m%d%H%M%S')}-{counter:04d}"
            db.insert(product_id, result)
            counter += 1
            cv2.imshow("Detection", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def mode_evaluate(args):
    from scripts.evaluate import evaluate
    evaluate(args.weights, args.data, args.imgsz)


def mode_benchmark(args):
    from scripts.benchmark import benchmark
    benchmark(args.config, args.frames)


def mode_stability(args):
    from scripts.stability_test import StabilityTester
    tester = StabilityTester(args.config, args.duration)
    tester.run()


def mode_api(args):
    from deploy.api_server import main as api_main
    import sys as _sys
    _sys.argv = [_sys.argv[0]]
    if args.host:
        _sys.argv += ["--host", args.host]
    if args.port:
        _sys.argv += ["--port", str(args.port)]
    api_main()


def main():
    parser = argparse.ArgumentParser(description="产品能效标签与缺陷检测系统")
    parser.add_argument("--config", default="config.yaml")
    subparsers = parser.add_subparsers(dest="mode")

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--resume", action="store_true")

    detect_parser = subparsers.add_parser("detect")
    detect_parser.add_argument("--camera", type=int, default=0)
    detect_parser.add_argument("--image")
    detect_parser.add_argument("--output")

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--weights", default="models/best.pt")
    eval_parser.add_argument("--data", default="data/dataset.yaml")
    eval_parser.add_argument("--imgsz", type=int, default=640)

    bench_parser = subparsers.add_parser("benchmark")
    bench_parser.add_argument("--frames", type=int, default=100)

    stab_parser = subparsers.add_parser("stability")
    stab_parser.add_argument("--duration", type=int, default=30)

    api_parser = subparsers.add_parser("api")
    api_parser.add_argument("--host", default="0.0.0.0")
    api_parser.add_argument("--port", type=int, default=5000)

    args = parser.parse_args()
    setup_logger()
    if args.mode is None:
        parser.print_help()
        return

    func = {
        "train": mode_train,
        "detect": mode_detect,
        "evaluate": mode_evaluate,
        "benchmark": mode_benchmark,
        "stability": mode_stability,
        "api": mode_api,
    }.get(args.mode)
    if func:
        func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
