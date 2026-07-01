#!/usr/bin/env python3
"""边缘设备主程序."""
import sys
import time
import argparse
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from inference.detector import EnergyLabelDetector
from utils.logger import setup_logger, get_logger
from utils.database import DetectionDatabase
from deploy.gpio_alert import GPIOAlert

logger = get_logger(__name__)


class EdgeDetectionApp:
    def __init__(self, config_path: str = "config.yaml"):
        self.detector = EnergyLabelDetector(config_path)
        self.db = DetectionDatabase()
        self.gpio = GPIOAlert(config_path)
        import yaml
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.camera_cfg = cfg.get("camera", {})

    def run(self):
        source = self.camera_cfg.get("source", 0)
        width = self.camera_cfg.get("width", 1920)
        height = self.camera_cfg.get("height", 1080)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.error("Failed to open camera.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, self.camera_cfg.get("fps", 30))
        logger.info(f"Camera opened: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        product_id_counter = 0
        frame_count = 0
        last_fps_time = time.perf_counter()
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Frame read failed")
                    continue
                product_id = f"EDGE-{time.strftime('%Y%m%d%H%M%S')}-{product_id_counter:04d}"
                product_id_counter += 1
                result = self.detector.detect(frame)
                if result["position_deviation"] or result["defects"]:
                    self.gpio.trigger()
                self.db.insert(product_id, result)
                annotated = self.detector.draw_results(frame, result)

                frame_count += 1
                t_now = time.perf_counter()
                if t_now - last_fps_time >= 1.0:
                    fps = frame_count / (t_now - last_fps_time)
                    frame_count = 0
                    last_fps_time = t_now
                    logger.info(f"FPS: {fps:.2f}")

                cv2.imshow("Edge Detection", annotated)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
        except KeyboardInterrupt:
            logger.info("Interrupted.")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.gpio.cleanup()
            logger.info("Application stopped.")


def main():
    parser = argparse.ArgumentParser(description="Edge Detection App")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    setup_logger()
    app = EdgeDetectionApp(args.config)
    app.run()


if __name__ == "__main__":
    main()
