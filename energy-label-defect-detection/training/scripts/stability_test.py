"""
长时间运行稳定性测试.
"""
import time
import argparse
import threading
import psutil
import numpy as np
from inference.detector import EnergyLabelDetector
from utils.logger import get_logger

logger = get_logger(__name__)


class StabilityTester:
    def __init__(self, config_path: str, duration_minutes: int = 30):
        self.config_path = config_path
        self.duration = duration_minutes * 60
        self.memory_samples = []
        self.fps_samples = []
        self.running = True
        self.error_count = 0

    def _monitor_memory(self):
        process = psutil.Process()
        while self.running:
            mem_mb = process.memory_info().rss / (1024 * 1024)
            self.memory_samples.append(mem_mb)
            time.sleep(1)

    def run(self):
        logger.info(f"Starting stability test (duration: {self.duration}s)...")
        detector = EnergyLabelDetector(self.config_path)
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        mem_thread = threading.Thread(target=self._monitor_memory, daemon=True)
        mem_thread.start()

        start_time = time.perf_counter()
        frame_count = 0
        last_fps_time = start_time
        last_fps_count = 0

        try:
            while True:
                elapsed = time.perf_counter() - start_time
                if elapsed >= self.duration:
                    break
                try:
                    detector.detect(test_image)
                    frame_count += 1
                except Exception as e:
                    self.error_count += 1
                    logger.error(f"Detection error: {e}")
                if elapsed - last_fps_time >= 1.0:
                    fps = (frame_count - last_fps_count) / (elapsed - last_fps_time)
                    self.fps_samples.append(fps)
                    last_fps_time = elapsed
                    last_fps_count = frame_count
        finally:
            self.running = False
            mem_thread.join(timeout=2)

        total_time = time.perf_counter() - start_time
        avg_fps = frame_count / total_time
        logger.info("=" * 60)
        logger.info("Stability Test Report")
        logger.info(f"  Total frames:  {frame_count}")
        logger.info(f"  Average FPS:   {avg_fps:.1f}")
        logger.info(f"  Errors:        {self.error_count}")
        if self.fps_samples:
            logger.info(f"  FPS min/avg/max: {min(self.fps_samples):.1f} / {np.mean(self.fps_samples):.1f} / {max(self.fps_samples):.1f}")
        if self.memory_samples:
            logger.info(f"  Memory min/avg/max (MB): {min(self.memory_samples):.1f} / {np.mean(self.memory_samples):.1f} / {max(self.memory_samples):.1f}")
            mem_growth = self.memory_samples[-1] - self.memory_samples[0]
            logger.info(f"  Memory growth: {mem_growth:+.1f} MB")
        logger.info("=" * 60)

        if avg_fps < 30:
            logger.warning("FPS below 30 - performance degradation detected!")
        if self.memory_samples:
            mem_growth_rate = (self.memory_samples[-1] - self.memory_samples[0]) / max(self.memory_samples[0], 1) * 100
            if mem_growth_rate > 10:
                logger.warning(f"Memory growth {mem_growth_rate:.1f}% exceeds 10% threshold!")


def main():
    parser = argparse.ArgumentParser(description="Stability test")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--duration", type=int, default=30, help="Duration in minutes")
    args = parser.parse_args()
    tester = StabilityTester(args.config, args.duration)
    tester.run()


if __name__ == "__main__":
    main()
