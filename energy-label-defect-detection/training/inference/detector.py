"""
推理检测器核心类.
"""
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import cv2
import yaml
from ultralytics import YOLO

from .postprocess import PostProcessor
from utils.logger import get_logger

logger = get_logger(__name__)


class EnergyLabelDetector:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        model_cfg = self.config["model"]
        infer_cfg = self.config["inference"]

        self.conf_threshold = infer_cfg["conf_threshold"]
        self.iou_threshold = infer_cfg["iou_threshold"]
        self.augment = infer_cfg.get("augment", False)
        self.input_size = tuple(model_cfg["input_size"])
        self.weights_path = model_cfg["weights_path"]

        device = infer_cfg.get("device", 0)
        if not Path(self.weights_path).exists():
            logger.warning(f"Weights not found at {self.weights_path}, using local pretrained model...")
            pretrained = model_cfg.get("pretrained_path", f"{model_cfg['architecture']}.pt")
            self.model = YOLO(pretrained)
        else:
            self.model = YOLO(self.weights_path)

        import torch as _torch
        if _torch.cuda.is_available():
            _torch.cuda.set_device(0)
        self.model.model.to("cuda:0" if _torch.cuda.is_available() else "cpu")
        self.class_names = self.config["class_names"]
        self.postprocessor = PostProcessor(self.config)
        self.box_id = self.config.get("box_id", 9)
        self.label_id = self.config.get("label_id", 8)

        # ── 两阶段推理: 加载 box-only 检测器 ──
        self.box_detector = None
        box_model_path = self.config.get("box_detector", {}).get("weights_path", "")
        # Resolve relative to config file directory
        config_dir = Path(config_path).resolve().parent
        if box_model_path and not Path(box_model_path).is_absolute():
            box_model_path = str(config_dir / box_model_path)
        if box_model_path and Path(box_model_path).exists():
            self.box_detector = YOLO(box_model_path)
            self.box_detector.model.to("cuda:0" if _torch.cuda.is_available() else "cpu")
            logger.info(f"Box detector loaded: {box_model_path}")
        else:
            logger.info("Box detector not found, using CV fallback only.")

        self._warmup()

    def _warmup(self):
        dummy = np.zeros((*self.input_size, 3), dtype=np.uint8)
        _ = self.model(dummy, verbose=False)
        logger.info("Model warmup complete.")


    def _find_box_cv(self, crop):
        """Locate energy rating box via edge detection on cropped label."""
        h, w = crop.shape[:2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        best = None
        best_score = 0
        
        # Try multiple adaptive threshold parameter sets
        for block_size in [31, 51, 71]:
            for C in [3, 5, 8]:
                thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, block_size, C)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    # Accept 1%-30% of crop area (real box ~3.6%, label boundary ~75%)
                    if area < 0.005 * w * h or area > 0.30 * w * h:
                        continue
                    
                    x, y, cw_box, ch = cv2.boundingRect(cnt)
                    aspect = cw_box / max(ch, 1)
                    if aspect < 0.3 or aspect > 3.0:
                        continue
                    
                    # Score: prefer rectangles with moderate area and aspect near 1
                    rect_score = area * (1.0 - abs(aspect - 1.0) / 2.0)
                    
                    # Penalize contours that are just the whole crop
                    coverage = area / (w * h)
                    if coverage > 0.8:
                        rect_score *= 0.1
                    
                    if rect_score > best_score:
                        best_score = rect_score
                        best = (x, y, cw_box, ch)
        
        return best
    def detect(self, image: np.ndarray, augment: bool = False) -> Dict:
        """Run detection on a single image.

        Args:
            image: BGR image (any size, resized internally).
            augment: If True, enable TTA (horizontal flip + multi-scale).
                     Improves recall at ~2-3× inference cost.
        """
        import torch
        t0 = time.perf_counter()
        results = self.model(
            image, conf=self.conf_threshold, iou=self.iou_threshold,
            imgsz=self.input_size[0], verbose=False, augment=augment,
        )
        
        # ── 两阶段推理: 在 label 区域用 box 检测器找 box ──
        if results[0].boxes is not None:
            classes = results[0].boxes.cls.int().tolist()
            label_indices = [i for i, c in enumerate(classes) if c == self.label_id]

            # Track box detections to add (defer tensor mod to avoid index shift)
            new_boxes = []

            # Only use two-stage when stage-1 missed box AND label confidence is OK
            stage1_has_box = 9 in classes
            stage1_box_conf = 0.0
            if stage1_has_box:
                box_indices = [i for i, c in enumerate(classes) if c == 9]
                stage1_box_conf = max(float(results[0].boxes.conf[bi]) for bi in box_indices)

            for label_idx in label_indices:
                # Safety: recompute bounds in case tensor was modified
                if label_idx >= len(results[0].boxes.xyxy):
                    continue
                x1, y1, x2, y2 = [int(v) for v in results[0].boxes.xyxy[label_idx].tolist()]
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                box_found = None

                # Skip two-stage if stage-1 already found any box (even low conf)
                if stage1_has_box:
                    continue

                # Stage 2a: Box detector (YOLO on label crop)
                if self.box_detector is not None:
                    box_results = self.box_detector(
                        crop, conf=0.08, iou=0.45, imgsz=320, verbose=False,
                    )
                    if box_results[0].boxes is not None and len(box_results[0].boxes) > 0:
                        best = box_results[0].boxes[0]
                        bx1, by1, bx2, by2 = [float(v) for v in best.xyxy[0].tolist()]
                        conf = float(best.conf[0])
                        box_found = (x1+bx1, y1+by1, x1+bx2, y1+by2, conf)

                # Stage 2b: CV fallback (always available as last resort)
                if box_found is None:
                    box_rect = self._find_box_cv(crop)
                    if box_rect is not None:
                        bx, by, bw, bh = box_rect
                        box_found = (x1+bx, y1+by, x1+bx+bw, y1+by+bh, 0.35)

                if box_found is not None:
                    new_boxes.append(box_found)

            # Add all found boxes at once
            if new_boxes:
                dev = results[0].boxes.data.device
                new_rows = torch.tensor(
                    [[b[0], b[1], b[2], b[3], b[4], 9.0] for b in new_boxes],
                    device=dev,
                )
                results[0].boxes.data = torch.cat([results[0].boxes.data, new_rows])

        t1 = time.perf_counter()
        inference_time_ms = (t1 - t0) * 1000.0
        result = self.postprocessor.process(results, image, inference_time_ms)
        return result

    def detect_batch(self, images: List[np.ndarray]) -> List[Dict]:
        return [self.detect(img) for img in images]

    def draw_results(self, image: np.ndarray, result: Dict) -> np.ndarray:
        return self.postprocessor.draw_annotations(image, result)
