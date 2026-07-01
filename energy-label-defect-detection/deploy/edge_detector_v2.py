#!/usr/bin/env python3
"""
边缘设备实时检测 v2 — 适用于 STM32N6 / 树莓派等边缘设备。
特性:
  - 三阶段推理: YOLO → Box detector → CV fallback
  - 缺陷类型标注在画面左下角
  - 检测到缺陷/位置偏差时触发 GPIO 报警
  - 结果存入 SQLite 数据库
  - 可选实时画面显示 (--display)

用法:
  python deploy/edge_detector_v2.py                     # 无显示, 仅日志+GPIO+DB
  python deploy/edge_detector_v2.py --display           # 带画面显示 (调试用)
  python deploy/edge_detector_v2.py --source 0          # 指定摄像头索引
  python deploy/edge_detector_v2.py --source video.mp4  # 测试视频文件
"""
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


class EdgeDetectionAppV2:
    """V2 版本: 缺陷左下角标注 + 实时告警."""

    def __init__(self, config_path: str = "config.yaml", show_display: bool = False):
        self.show_display = show_display
        self.detector = EnergyLabelDetector(config_path)
        self.db = DetectionDatabase()
        self.gpio = GPIOAlert(config_path)

        import yaml
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.camera_cfg = cfg.get("camera", {})
        self.alert_interval_s = self.camera_cfg.get("alert_interval", 1.0)
        self.last_alert_time = 0.0

    def _should_alert(self) -> bool:
        """防止告警风暴: 至少间隔 alert_interval_s 秒."""
        now = time.perf_counter()
        if now - self.last_alert_time >= self.alert_interval_s:
            self.last_alert_time = now
            return True
        return False

    def run(self, source=None):
        # ── 打开视频源 ──
        if source is None:
            source = self.camera_cfg.get("source", 0)

        # 判断是摄像头索引还是文件路径
        if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
            source = int(source) if isinstance(source, str) else source
            cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(str(source))

        if not cap.isOpened():
            logger.error(f"Failed to open source: {source}")
            return

        width = self.camera_cfg.get("width", 1280)
        height = self.camera_cfg.get("height", 720)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, self.camera_cfg.get("fps", 30))

        actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        logger.info(f"Source opened: {actual_w:.0f}x{actual_h:.0f}")

        frame_count = 0
        product_id_counter = 0
        last_fps_time = time.perf_counter()
        window_name = "Edge Detection V2"

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Frame read failed, retrying...")
                    time.sleep(0.01)
                    continue

                product_id = f"EDGE-{time.strftime('%Y%m%d%H%M%S')}-{product_id_counter:04d}"
                product_id_counter += 1

                # ── 推理 ──
                result = self.detector.detect(frame)

                # ── 告警判断 ──
                has_issue = result["position_deviation"] or len(result.get("defects", [])) > 0

                if has_issue:
                    defects_str = ", ".join(
                        [d["defect_type"] for d in result.get("defects", [])]
                    ) or "none"
                    dev_str = "DEVIATED" if result["position_deviation"] else "OK"
                    logger.warning(
                        f"ISSUE DETECTED | ID={product_id} | "
                        f"Position={dev_str} (dx={result['offset_x']:.3f}, dy={result['offset_y']:.3f}) | "
                        f"Defects=[{defects_str}]"
                    )
                    if self._should_alert():
                        self.gpio.trigger()

                # ── 数据库记录 ──
                self.db.insert(product_id, result)

                # ── 绘制 ──
                if self.show_display:
                    annotated = self.detector.postprocessor.draw_v2(frame, result)
                    # 叠加产品 ID
                    cv2.putText(annotated, f"ID: {product_id}",
                                (10, annotated.shape[0] - 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                    cv2.imshow(window_name, annotated)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        logger.info("User quit via display.")
                        break

                # ── FPS 日志 ──
                frame_count += 1
                t_now = time.perf_counter()
                if t_now - last_fps_time >= 5.0:
                    fps = frame_count / (t_now - last_fps_time)
                    frame_count = 0
                    last_fps_time = t_now
                    logger.info(f"FPS: {fps:.2f} | Total processed: {product_id_counter}")

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            cap.release()
            if self.show_display:
                cv2.destroyAllWindows()
            self.gpio.cleanup()
            logger.info("Application stopped.")


def main():
    parser = argparse.ArgumentParser(description="Edge Detection App V2")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--display", action="store_true", help="启用实时画面显示")
    parser.add_argument("--source", default=None,
                        help="视频源: 摄像头索引 (0,1,...) 或视频文件路径")
    args = parser.parse_args()

    setup_logger()
    app = EdgeDetectionAppV2(args.config, show_display=args.display)

    source = args.source
    if source is not None and source.isdigit():
        source = int(source)

    app.run(source=source)


if __name__ == "__main__":
    main()
