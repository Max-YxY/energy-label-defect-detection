"""
树莓派 ONNX 推理检测器 — 与原版 EnergyLabelDetector 接口一致
使用 ONNX Runtime 替代 Ultralytics YOLO，保持相同算法逻辑
"""
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import cv2
import yaml
import onnxruntime as ort

from .postprocess import PostProcessor
from utils.logger import get_logger

logger = get_logger(__name__)


def nms(boxes, scores, iou_threshold):
    """非极大值抑制，与原版一致"""
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-10)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def postprocess_yolo(output, img_shape, conf_threshold, iou_threshold):
    """
    YOLOv8 ONNX 输出后处理
    output: (1, num_classes+4, num_anchors) → 同原版 YOLO 输出格式
    """
    predictions = np.squeeze(output).T  # (num_anchors, 4+num_classes)
    boxes_center = predictions[:, :4]
    scores = predictions[:, 4:]

    max_scores = np.max(scores, axis=1)
    class_ids = np.argmax(scores, axis=1)

    valid = max_scores >= conf_threshold
    if not np.any(valid):
        return [], [], []

    boxes_center = boxes_center[valid]
    max_scores = max_scores[valid]
    class_ids = class_ids[valid]

    # 中心点 → 左上角+右下角
    xc, yc, w, h = boxes_center[:, 0], boxes_center[:, 1], boxes_center[:, 2], boxes_center[:, 3]
    x1 = np.clip(xc - w / 2, 0, 1)
    y1 = np.clip(yc - h / 2, 0, 1)
    x2 = np.clip(xc + w / 2, 0, 1)
    y2 = np.clip(yc + h / 2, 0, 1)

    boxes = np.stack([x1, y1, x2, y2], axis=1)
    keep = nms(boxes, max_scores, iou_threshold)

    if len(keep) == 0:
        return [], [], []

    final_boxes = boxes[keep]
    final_scores = max_scores[keep]
    final_class_ids = class_ids[keep]

    # 缩放到原始图像尺寸
    h_img, w_img = img_shape[:2]
    final_boxes[:, [0, 2]] *= w_img
    final_boxes[:, [1, 3]] *= h_img
    final_boxes = final_boxes.astype(np.int32)

    return final_boxes, final_scores, final_class_ids


def find_box_cv(crop):
    """自适应阈值边缘检测找 box，与原版 _find_box_cv 一致"""
    h, w = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    best = None
    best_score = 0

    for block_size in [31, 51, 71]:
        for C in [3, 5, 8]:
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, block_size, C)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < 0.005 * w * h or area > 0.30 * w * h:
                    continue
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect = cw / max(ch, 1)
                if aspect < 0.3 or aspect > 3.0:
                    continue
                rect_score = area * (1.0 - abs(aspect - 1.0) / 2.0)
                coverage = area / (w * h)
                if coverage > 0.8:
                    rect_score *= 0.1
                if rect_score > best_score:
                    best_score = rect_score
                    best = (x, y, cw, ch)
    return best


class EnergyLabelDetector:
    """
    树莓派 ONNX 版检测器 — 接口与原版完全一致
    
    用法:
        detector = EnergyLabelDetector("config.yaml")
        result = detector.detect(image)  # image: BGR numpy array
        annotated = detector.draw_results(image, result)
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        model_cfg = self.config["model"]
        infer_cfg = self.config["inference"]

        self.conf_threshold = infer_cfg["conf_threshold"]
        self.iou_threshold = infer_cfg["iou_threshold"]
        self.input_size = tuple(model_cfg["input_size"])
        self.weights_path = model_cfg["weights_path"]

        # ── 加载 ONNX 主模型 ──
        onnx_path = str(Path(self.weights_path).with_suffix(".onnx"))
        if not Path(onnx_path).exists():
            # 尝试 best_320.onnx / best_416.onnx 等变体
            for candidate in [
                str(Path(self.weights_path).parent / f"best_{self.input_size[0]}.onnx"),
                str(Path(self.weights_path).with_suffix(".onnx")),
            ]:
                if Path(candidate).exists():
                    onnx_path = candidate
                    break

        logger.info(f"Loading ONNX model: {onnx_path}")
        so = ort.SessionOptions()
        so.enable_cpu_mem_arena = True
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            onnx_path, sess_options=so, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

        self.class_names = self.config["class_names"]
        self.postprocessor = PostProcessor(self.config)
        self.box_id = self.config.get("box_id", 9)
        self.label_id = self.config.get("label_id", 8)

        # ── 两阶段推理: 加载 box-only ONNX 检测器 ──
        self.box_session = None
        self.box_input_name = None
        box_model_cfg = self.config.get("box_detector", {})
        box_model_path = box_model_cfg.get("weights_path", "")
        config_dir = Path(config_path).resolve().parent
        if box_model_path and not Path(box_model_path).is_absolute():
            box_model_path = str(config_dir / box_model_path)
        if box_model_path:
            box_onnx = str(Path(box_model_path).with_suffix(".onnx"))
            if Path(box_onnx).exists():
                box_so = ort.SessionOptions()
                box_so.enable_cpu_mem_arena = True
                box_so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.box_session = ort.InferenceSession(
                    box_onnx, sess_options=box_so, providers=["CPUExecutionProvider"]
                )
                self.box_input_name = self.box_session.get_inputs()[0].name
                logger.info(f"Box detector loaded: {box_onnx}")
            else:
                logger.info(f"Box detector ONNX not found: {box_onnx}, using CV fallback only.")
        else:
            logger.info("Box detector not configured, using CV fallback only.")

        self._warmup()

    def _warmup(self):
        dummy = np.zeros((*self.input_size, 3), dtype=np.uint8)
        self._infer_onnx(dummy)
        logger.info("Model warmup complete.")

    def _infer_onnx(self, image: np.ndarray) -> np.ndarray:
        """ONNX 推理：预处理 + run"""
        input_img = cv2.resize(image, self.input_size)
        input_tensor = input_img.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        return outputs[0]

    def _infer_box_onnx(self, crop: np.ndarray) -> List:
        """Box 检测器 ONNX 推理"""
        if self.box_session is None:
            return []
        crop_resized = cv2.resize(crop, (320, 320))
        input_tensor = crop_resized.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        outputs = self.box_session.run(None, {self.box_input_name: input_tensor})
        # box_detector 输出 [1, 5, 2100]
        pred = np.squeeze(outputs[0]).T  # (2100, 5)
        box_confs = pred[:, 4]
        if len(box_confs) == 0:
            return []
        best_idx = np.argmax(box_confs)
        if box_confs[best_idx] >= 0.08:
            bx, by, bw, bh = pred[best_idx, :4]
            return [(bx, by, bw, bh, float(box_confs[best_idx]))]
        return []

    def detect(self, image: np.ndarray, augment: bool = False) -> Dict:
        """单图检测 — 接口与原版一致"""
        t0 = time.perf_counter()
        oh, ow = image.shape[:2]

        # ── Stage 1: 主模型推理 ──
        output = self._infer_onnx(image)
        boxes, scores, class_ids = postprocess_yolo(
            output, (oh, ow), self.conf_threshold, self.iou_threshold
        )

        # 组装检测结果（与原版 results[0].boxes 格式一致）
        all_boxes = []
        for box, score, cls_id in zip(boxes, scores, class_ids):
            all_boxes.append({
                "class_id": int(cls_id),
                "class_name": self.class_names.get(int(cls_id), "unknown"),
                "confidence": round(float(score), 4),
                "bbox": [round(float(v), 2) for v in box],
            })

        # ── 两阶段推理: 在 label 区域用 box 检测器找 box ──
        class_id_list = [b["class_id"] for b in all_boxes]
        stage1_has_box = self.box_id in class_id_list

        if not stage1_has_box:
            label_indices = [i for i, b in enumerate(all_boxes) if b["class_id"] == self.label_id]
            for idx in label_indices:
                bbox = all_boxes[idx]["bbox"]
                x1, y1, x2, y2 = [int(v) for v in bbox]
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                box_found = None

                # Stage 2a: Box detector ONNX on crop
                box_results = self._infer_box_onnx(crop)
                if box_results:
                    bx, by, bw, bh, conf = box_results[0]
                    ch, cw = crop.shape[:2]
                    bx1_c = int((bx - bw / 2) * cw)
                    by1_c = int((by - bh / 2) * ch)
                    bx2_c = int((bx + bw / 2) * cw)
                    by2_c = int((by + bh / 2) * ch)
                    box_found = (x1 + bx1_c, y1 + by1_c,
                                 x1 + bx2_c, y1 + by2_c, conf)

                # Stage 2b: CV fallback
                if box_found is None:
                    rect = find_box_cv(crop)
                    if rect is not None:
                        bx, by, bw, bh = rect
                        box_found = (x1 + bx, y1 + by,
                                     x1 + bx + bw, y1 + by + bh, 0.35)

                if box_found is not None:
                    all_boxes.append({
                        "class_id": self.box_id,
                        "class_name": self.class_names.get(self.box_id, "box"),
                        "confidence": round(box_found[4], 4),
                        "bbox": [round(box_found[0], 2), round(box_found[1], 2),
                                 round(box_found[2], 2), round(box_found[3], 2)],
                    })

        t1 = time.perf_counter()
        inference_time_ms = (t1 - t0) * 1000.0

        # ── 直接计算结果（跳过 postprocessor.process 的 YOLO 对象依赖）──
        energy_level = self.postprocessor._extract_energy_level(all_boxes)
        defects = self.postprocessor._extract_defects(all_boxes)

        # 位置偏差
        position_deviation = False
        offset_x = 0.0
        offset_y = 0.0
        label_boxes = [b for b in all_boxes if b["class_id"] == self.label_id]
        box_boxes = [b for b in all_boxes if b["class_id"] == self.box_id]
        pos_conf_th = self.postprocessor.position_cfg.get("confidence_threshold", 0.3)
        label_conf = label_boxes[0]["confidence"] if label_boxes else 0.0
        box_conf = box_boxes[0]["confidence"] if box_boxes else 0.0
        if label_boxes and box_boxes and label_conf >= pos_conf_th and box_conf >= pos_conf_th:
            lb = label_boxes[0]["bbox"]
            bb = box_boxes[0]["bbox"]
            lcx = (lb[0] + lb[2]) / 2.0
            lcy = (lb[1] + lb[3]) / 2.0
            bcx = (bb[0] + bb[2]) / 2.0
            bcy = (bb[1] + bb[3]) / 2.0
            bw = bb[2] - bb[0]
            bh = bb[3] - bb[1]
            if bw > 0 and bh > 0:
                ox = (lcx - bcx) / bw
                oy = (lcy - bcy) / bh
                tx = self.postprocessor.position_cfg.get("tolerance_x", 0.18)
                ty = self.postprocessor.position_cfg.get("tolerance_y", 0.16)
                if abs(ox) > tx or abs(oy) > ty:
                    position_deviation = True
                else:
                    margins = [
                        abs((lb[0] - bb[0]) / bw),
                        abs((bb[2] - lb[2]) / bw),
                        abs((lb[1] - bb[1]) / bh),
                        abs((bb[3] - lb[3]) / bh),
                    ]
                    if min(margins) < 0.01:
                        position_deviation = True
                offset_x = ox
                offset_y = oy

        result = {
            "energy_level": energy_level,
            "defects": defects,
            "position_deviation": position_deviation,
            "offset_x": round(offset_x, 4),
            "offset_y": round(offset_y, 4),
            "boxes": all_boxes,
            "inference_time_ms": round(inference_time_ms, 2),
        }
        return result

    def detect_batch(self, images: List[np.ndarray]) -> List[Dict]:
        return [self.detect(img) for img in images]

    def draw_results(self, image: np.ndarray, result: Dict) -> np.ndarray:
        return self.postprocessor.draw_annotations(image, result)

    def draw_v2(self, image: np.ndarray, result: Dict) -> np.ndarray:
        return self.postprocessor.draw_v2(image, result)
