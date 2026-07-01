#!/usr/bin/env python3
"""
STM32N647 边缘设备 — 实时检测模块（可导入）

基于 webcam_demo.py 的摄像头调用模式改造，面向 STM32N6 部署。
与 PC 版的唯一区别：增加了 GPIO 报警 + 数据库记录 + 无头模式。

用法:
  # 作为模块导入
  from stm32_edge_detect import STM32EdgeDetector
  detector = STM32EdgeDetector()
  detector.run(camera_index=0, show_display=True)

  # 命令行
  python stm32_edge_detect.py --display
  python stm32_edge_detect.py --source 0 --no-gpio
"""

import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import cv2

# 确保项目根在 path 中
_PROJECT = Path(__file__).resolve().parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

from inference.detector import EnergyLabelDetector
from utils.logger import setup_logger, get_logger
from utils.database import DetectionDatabase

logger = get_logger(__name__)


class STM32EdgeDetector:
    """
    STM32N647 边缘设备检测器 — 可直接导入使用。

    与 PC 版 webcam_demo.py 的摄像头调用方式完全一致 (cv2.VideoCapture),
    额外集成:
      - SQLite 数据库实时记录
      - GPIO 报警 (RPi.GPIO / pyA20, 可禁用)
      - 跳帧优化 (降低 CPU 负载, 适配 600GOPS NPU)
      - FPS 日志
      - 无头模式 (--display 开关)

    使用:
        det = STM32EdgeDetector("config.yaml")
        det.run(camera_index=0, show_display=True)
        # 或
        result = det.detect_one_frame(frame)
    """

    # ── 跳帧参数 (边缘设备优化) ──
    SKIP_FRAMES = 2       # 每 N 帧推理一次
    SMOOTH_ALPHA = 0.3    # 框平滑系数

    def __init__(self, config_path: str = "config.yaml",
                 enable_gpio: bool = True,
                 enable_db: bool = True):
        """
        参数:
            config_path: config.yaml 路径
            enable_gpio: 启用 GPIO 报警 (树莓派/香橙派上设为 True)
            enable_db: 启用 SQLite 数据库记录
        """
        self.config_path = config_path
        self.enable_gpio = enable_gpio
        self.enable_db = enable_db

        # ── 加载检测器 (三阶段管线) ──
        logger.info("加载模型...")
        self.detector = EnergyLabelDetector(config_path)

        # ── 数据库 ──
        self.db = DetectionDatabase() if enable_db else None

        # ── GPIO ──
        self.gpio = None
        self._last_alert_time = 0.0
        self._alert_interval = 1.0
        if enable_gpio:
            try:
                from deploy.gpio_alert import GPIOAlert
                self.gpio = GPIOAlert(config_path)
                if self.gpio.enabled:
                    logger.info("GPIO 报警已启用")
                else:
                    logger.info("GPIO 未启用 (无GPIO库或配置关闭)")
            except Exception as e:
                logger.warning(f"GPIO 初始化失败: {e}")

        # ── 摄像头配置 ──
        import yaml
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self._cam_cfg = cfg.get("camera", {})

        # ── 状态 ──
        self._running = False
        self._frame_idx = 0
        self._last_result: Optional[Dict] = None
        self._last_boxes: Optional[list] = None

    # ==================================================================
    # 核心 API
    # ==================================================================

    def detect_one_frame(self, frame) -> Dict[str, Any]:
        """
        对单帧执行三阶段检测。
        与 webcam_demo.py 内部调用完全一致。
        """
        return self.detector.detect(frame)

    def run(self, camera_index: Optional[int] = None,
            show_display: bool = False,
            source: Optional[str] = None):
        """
        启动实时检测循环。

        参数:
            camera_index: PC 摄像头索引 (0, 1, ...)
            show_display: 是否显示 OpenCV 窗口
            source: 视频文件路径 (覆盖 camera_index)
        """
        # ── 打开摄像头 (与 webcam_demo.py 完全相同的方式) ──
        if source is not None:
            cap = cv2.VideoCapture(source)
        elif camera_index is not None:
            cap = cv2.VideoCapture(camera_index)
        else:
            idx = self._cam_cfg.get("source", 0)
            cap = cv2.VideoCapture(idx)

        if not cap.isOpened():
            # 尝试备用索引
            for idx in [0, 1, 2]:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    break
            if not cap.isOpened():
                logger.error("无法打开摄像头！")
                return

        width = self._cam_cfg.get("width", 640)
        height = self._cam_cfg.get("height", 480)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, self._cam_cfg.get("fps", 30))

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"摄像头已打开: {actual_w}x{actual_h}")

        self._running = True
        self._frame_idx = 0
        self._last_result = None
        self._last_boxes = None

        product_counter = 0
        real_fps_count = 0
        last_fps_time = time.perf_counter()
        fps_display = 0.0
        window_name = "STM32 Edge Detection"

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                real_fps_count += 1

                # ── 跳帧推理 (与 webcam_demo.py 相同策略) ──
                if self._frame_idx % (self.SKIP_FRAMES + 1) == 0:
                    result = self.detect_one_frame(frame)

                    # 平滑检测框
                    if self._last_boxes is not None:
                        result["boxes"] = self._smooth_boxes(
                            self._last_boxes, result["boxes"]
                        )
                    self._last_result = result
                    self._last_boxes = result["boxes"]
                else:
                    result = self._last_result

                self._frame_idx += 1

                if result is None:
                    continue

                # ── 产品 ID ──
                product_id = f"EDGE-{time.strftime('%Y%m%d%H%M%S')}-{product_counter:04d}"
                product_counter += 1

                # ── 告警判断 ──
                has_issue = (result["position_deviation"] or
                             len(result.get("defects", [])) > 0)
                if has_issue:
                    self._log_issue(product_id, result)
                    self._trigger_alert()

                # ── 数据库记录 ──
                if self.db is not None:
                    self.db.insert(product_id, result)

                # ── 显示 (与 webcam_demo.py 相同) ──
                if show_display:
                    annotated = self.detector.postprocessor.draw_v2(frame, result)
                    # 叠加 FPS
                    cv2.putText(annotated, f"Real: {fps_display:.1f} FPS",
                                (annotated.shape[1] - 170,
                                 annotated.shape[0] - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 255, 0), 2)
                    cv2.putText(annotated, f"ID: {product_id}",
                                (10, annotated.shape[0] - 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (200, 200, 200), 1)

                    cv2.imshow(window_name, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        logger.info("用户退出。")
                        break

                # ── FPS 日志 ──
                t_now = time.perf_counter()
                if t_now - last_fps_time >= 1.0:
                    fps_display = real_fps_count / (t_now - last_fps_time)
                    real_fps_count = 0
                    last_fps_time = t_now
                if t_now - last_fps_time >= 5.0:
                    logger.info(f"FPS: {fps_display:.1f} | "
                                f"累计: {product_counter}")

        except KeyboardInterrupt:
            logger.info("中断。")
        finally:
            cap.release()
            if show_display:
                cv2.destroyAllWindows()
            if self.gpio is not None:
                self.gpio.cleanup()
            logger.info(f"已停止。共处理 {product_counter} 帧。")

    def stop(self):
        """外部停止检测循环。"""
        self._running = False

    # ==================================================================
    # 内部方法
    # ==================================================================

    def _smooth_boxes(self, old_boxes, new_boxes):
        """对检测框做指数平滑，减少抖动 (与 webcam_demo.py 相同)。"""
        alpha = self.SMOOTH_ALPHA
        if not old_boxes or not new_boxes:
            return new_boxes
        smoothed = []
        new_matched = set()
        for old in old_boxes:
            best, best_dist = None, float("inf")
            for j, new in enumerate(new_boxes):
                if j in new_matched:
                    continue
                if old["class_id"] != new["class_id"]:
                    continue
                ocx = (old["bbox"][0] + old["bbox"][2]) / 2
                ocy = (old["bbox"][1] + old["bbox"][3]) / 2
                ncx = (new["bbox"][0] + new["bbox"][2]) / 2
                ncy = (new["bbox"][1] + new["bbox"][3]) / 2
                dist = ((ocx - ncx) ** 2 + (ocy - ncy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best = j
            if best is not None and best_dist < 50:
                new_matched.add(best)
                ob, nb = old["bbox"], new_boxes[best]["bbox"]
                smoothed.append({
                    **new_boxes[best],
                    "bbox": [
                        ob[0] * (1 - alpha) + nb[0] * alpha,
                        ob[1] * (1 - alpha) + nb[1] * alpha,
                        ob[2] * (1 - alpha) + nb[2] * alpha,
                        ob[3] * (1 - alpha) + nb[3] * alpha,
                    ],
                    "confidence": (old["confidence"] +
                                   new_boxes[best]["confidence"]) / 2,
                })
        for j, new in enumerate(new_boxes):
            if j not in new_matched:
                smoothed.append(new)
        return smoothed

    def _log_issue(self, product_id: str, result: Dict):
        defects_str = ", ".join(
            [d["defect_type"] for d in result.get("defects", [])]
        ) or "none"
        dev_str = "DEVIATED" if result["position_deviation"] else "OK"
        logger.warning(
            f"⚠️  ISSUE | ID={product_id} | "
            f"Position={dev_str} (dx={result['offset_x']:.3f}, "
            f"dy={result['offset_y']:.3f}) | "
            f"Defects=[{defects_str}]"
        )

    def _trigger_alert(self):
        """触发 GPIO 报警 (带防抖)。"""
        now = time.perf_counter()
        if now - self._last_alert_time >= self._alert_interval:
            self._last_alert_time = now
            if self.gpio is not None:
                self.gpio.trigger()


# ======================================================================
# 命令行入口
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="STM32N647 边缘检测器 — 基于 webcam_demo.py 改造"
    )
    parser.add_argument("--config", default="config.yaml",
                        help="配置文件路径")
    parser.add_argument("--display", action="store_true",
                        help="启用实时画面显示")
    parser.add_argument("--source", default=None,
                        help="摄像头索引 (0,1,...) 或视频文件路径")
    parser.add_argument("--no-gpio", action="store_true",
                        help="禁用 GPIO 报警")
    parser.add_argument("--no-db", action="store_true",
                        help="禁用数据库记录")

    args = parser.parse_args()
    setup_logger()

    det = STM32EdgeDetector(
        config_path=args.config,
        enable_gpio=not args.no_gpio,
        enable_db=not args.no_db,
    )

    source = args.source
    if source is not None and source.isdigit():
        source = int(source)
    camera_index = source if isinstance(source, int) else None

    det.run(camera_index=camera_index,
            show_display=args.display,
            source=source if isinstance(source, str) else None)


if __name__ == "__main__":
    main()
