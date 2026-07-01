"""
摄像头抽象层.
"""
from typing import Optional, Tuple
import cv2
import numpy as np
from utils.logger import get_logger

logger = get_logger(__name__)


class Camera:
    def __init__(self, source: int = 0, width: int = 1920, height: int = 1080, fps: int = 30, api_preference: str = "CAP_V4L2"):
        self.source = source
        self.width = width
        self.height = height
        self.target_fps = fps

        api_map = {"CAP_V4L2": cv2.CAP_V4L2, "CAP_ANY": cv2.CAP_ANY, "CAP_GSTREAMER": cv2.CAP_GSTREAMER}
        api = api_map.get(api_preference, cv2.CAP_V4L2)
        self.cap = cv2.VideoCapture(source, api)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(source)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open camera source: {source}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        logger.info(f"Camera initialized: {actual_w}x{actual_h}")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        return self.cap.read()

    def release(self):
        if self.cap is not None:
            self.cap.release()
            logger.info("Camera released.")

    def is_opened(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
