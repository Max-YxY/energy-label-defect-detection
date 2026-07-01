"""
性能基准测试脚本.
"""
import time
import argparse
import numpy as np
from inference.detector import EnergyLabelDetector
from utils.logger import get_logger

logger = get_logger(__name__)


def benchmark(config_path: str, num_frames: int = 100, warmup: int = 10):
    detector = EnergyLabelDetector(config_path)
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    logger.info(f"Warming up ({warmup} frames)...")
    for _ in range(warmup):
        detector.detect(test_image)

    logger.info(f"Benchmarking ({num_frames} frames)...")
    latencies = []
    for i in range(num_frames):
        t0 = time.perf_counter()
        detector.detect(test_image)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)
        if (i + 1) % 50 == 0:
            logger.info(f"  Progress: {i + 1}/{num_frames}")

    latencies = np.array(latencies)
    avg_ms = np.mean(latencies)
    fps = 1000.0 / avg_ms
    logger.info("=" * 60)
    logger.info(f"  Average latency: {avg_ms:.2f} ms")
    logger.info(f"  Std deviation:   {np.std(latencies):.2f} ms")
    logger.info(f"  FPS:             {fps:.1f}")
    logger.info("=" * 60)
    return {"avg_ms": avg_ms, "fps": fps}


def main():
    parser = argparse.ArgumentParser(description="Inference benchmark")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--frames", type=int, default=100)
    args = parser.parse_args()
    benchmark(args.config, args.frames)


if __name__ == "__main__":
    main()
