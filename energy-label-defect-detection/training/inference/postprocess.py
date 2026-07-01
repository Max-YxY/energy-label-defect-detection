"""
后处理模块.
"""
from typing import Dict, List, Optional

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class PostProcessor:
    def __init__(self, config: Dict):
        self.config = config
        self.class_names = config["class_names"]
        self.energy_level_ids = set(config["energy_level_ids"])
        self.defect_ids = set(config["defect_ids"])
        self.label_id = config["label_id"]
        self.box_id = config["box_id"]
        self.position_cfg = config["position_deviation"]

        self.colors = {
            "level": (0, 255, 0),
            "defect": (0, 0, 255),
            "label": (255, 255, 0),
            "box": (255, 0, 0),
        }

    def process(self, results, image: np.ndarray, inference_time_ms: float) -> Dict:
        h, w = image.shape[:2]
        all_boxes = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes_data = results[0].boxes
            for box in boxes_data:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                all_boxes.append({
                    "class_id": cls_id,
                    "class_name": self.class_names.get(cls_id, "unknown"),
                    "confidence": round(conf, 4),
                    "bbox": [round(v, 2) for v in xyxy],
                    "bbox_normalized": [
                        round(xyxy[0] / w, 4),
                        round(xyxy[1] / h, 4),
                        round((xyxy[2] - xyxy[0]) / w, 4),
                        round((xyxy[3] - xyxy[1]) / h, 4),
                    ],
                })

        energy_level = self._extract_energy_level(all_boxes)
        defects = self._extract_defects(all_boxes)

        label_boxes = [b for b in all_boxes if b["class_id"] == self.label_id]
        box_boxes = [b for b in all_boxes if b["class_id"] == self.box_id]

                # ── Position Deviation (center offset + edge margin) ──
        position_deviation = False
        offset_x = 0.0
        offset_y = 0.0

        # 置信度门控: label/box 置信度过低时跳过位置偏差判断 → 降低正常集 FP
        pos_conf_th = self.position_cfg.get("confidence_threshold", 0.3)
        label_conf = label_boxes[0]["confidence"] if label_boxes else 0.0
        box_conf = box_boxes[0]["confidence"] if box_boxes else 0.0

        if label_boxes and box_boxes and label_conf >= pos_conf_th and box_conf >= pos_conf_th:
            label_bbox = label_boxes[0]["bbox"]
            box_bbox = box_boxes[0]["bbox"]
            label_cx = (label_bbox[0] + label_bbox[2]) / 2.0
            label_cy = (label_bbox[1] + label_bbox[3]) / 2.0
            box_cx = (box_bbox[0] + box_bbox[2]) / 2.0
            box_cy = (box_bbox[1] + box_bbox[3]) / 2.0
            box_w = box_bbox[2] - box_bbox[0]
            box_h = box_bbox[3] - box_bbox[1]

            if box_w > 0 and box_h > 0:
                offset_x = (label_cx - box_cx) / box_w
                offset_y = (label_cy - box_cy) / box_h
                tolerance_x = self.position_cfg.get("tolerance_x", 0.18)
                tolerance_y = self.position_cfg.get("tolerance_y", 0.16)

                # Method 1: center offset
                if abs(offset_x) > tolerance_x or abs(offset_y) > tolerance_y:
                    position_deviation = True
                else:
                    # Method 2: edge margin — label edge touches box edge
                    left   = abs((label_bbox[0] - box_bbox[0]) / box_w)
                    right  = abs((box_bbox[2] - label_bbox[2]) / box_w)
                    top    = abs((label_bbox[1] - box_bbox[1]) / box_h)
                    bottom = abs((box_bbox[3] - label_bbox[3]) / box_h)
                    if min(left, right, top, bottom) < 0.01:
                        position_deviation = True

        return {
            "energy_level": energy_level,
            "defects": defects,
            "position_deviation": position_deviation,
            "offset_x": round(offset_x, 4),
            "offset_y": round(offset_y, 4),
            "boxes": all_boxes,
            "inference_time_ms": round(inference_time_ms, 2),
        }

    def _extract_energy_level(self, boxes: List[Dict]) -> Optional[int]:
        energy_boxes = [b for b in boxes if b["class_id"] in self.energy_level_ids]
        if not energy_boxes:
            return None
        energy_boxes.sort(key=lambda b: b["confidence"], reverse=True)
        return energy_boxes[0]["class_id"] + 1

    def _extract_defects(self, boxes: List[Dict]) -> List[Dict]:
        defect_boxes = [b for b in boxes if b["class_id"] in self.defect_ids]
        seen_classes = set()
        unique_defects = []
        defect_boxes.sort(key=lambda b: b["confidence"], reverse=True)
        for box in defect_boxes:
            if box["class_id"] not in seen_classes:
                unique_defects.append({
                    "defect_type": box["class_name"],
                    "confidence": box["confidence"],
                    "bbox": box["bbox"],
                })
                seen_classes.add(box["class_id"])
        return unique_defects

    def draw_annotations(self, image: np.ndarray, result: Dict) -> np.ndarray:
        img = image.copy()
        h, w = img.shape[:2]

        for box in result["boxes"]:
            cls_id = box["class_id"]
            bbox = box["bbox"]
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cls_name = box["class_name"]
            conf = box["confidence"]

            if cls_id in self.energy_level_ids:
                color = self.colors["level"]
            elif cls_id in self.defect_ids:
                color = self.colors["defect"]
            elif cls_id == self.label_id:
                color = self.colors["label"]
            elif cls_id == self.box_id:
                color = self.colors["box"]
            else:
                color = (128, 128, 128)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label_text = f"{cls_name} {conf:.2f}"
            cv2.putText(img, label_text, (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        energy_level = result.get("energy_level")
        if energy_level is not None:
            text = f"Energy Level: {energy_level}"
            cv2.putText(img, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
        else:
            cv2.putText(img, "Energy Level: UNKNOWN", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        defects = result.get("defects", [])
        y_offset = 80
        if defects:
            for d in defects:
                text = f"Defect: {d['defect_type']} ({d['confidence']:.2f})"
                cv2.putText(img, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                y_offset += 30
        else:
            cv2.putText(img, "Defects: None", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30

        deviated = result.get("position_deviation", False)
        offset_x = result.get("offset_x", 0.0)
        offset_y = result.get("offset_y", 0.0)
        dev_text = f"Position: {'DEVIATED' if deviated else 'OK'} (dx={offset_x:.3f}, dy={offset_y:.3f})"
        dev_color = (0, 0, 255) if deviated else (0, 255, 0)
        cv2.putText(img, dev_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, dev_color, 2)
        y_offset += 35

        inf_time = result.get("inference_time_ms", 0)
        fps = 1000.0 / inf_time if inf_time > 0 else 0
        cv2.putText(img, f"Inference: {inf_time:.1f}ms / {fps:.1f} FPS",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return img

    def draw_v2(self, image: np.ndarray, result: Dict) -> np.ndarray:
        """
        V2 绘制布局:
          左上: Energy Level + Position Deviation 状态
          左下: 缺陷类型列表 (stain / damage / wrinkle)
          右下: FPS
          检测框保留在原位置
        """
        img = image.copy()
        h, w = img.shape[:2]

        # ── 绘制所有检测框 ──
        for box in result["boxes"]:
            cls_id = box["class_id"]
            bbox = box["bbox"]
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cls_name = box["class_name"]
            conf = box["confidence"]

            if cls_id in self.energy_level_ids:
                color = self.colors["level"]
            elif cls_id in self.defect_ids:
                color = self.colors["defect"]
            elif cls_id == self.label_id:
                color = self.colors["label"]
            elif cls_id == self.box_id:
                color = self.colors["box"]
            else:
                color = (128, 128, 128)

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label_text = f"{cls_name} {conf:.2f}"
            cv2.putText(img, label_text, (x1, max(y1 - 8, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # ── 左上: 能效等级 ──
        energy_level = result.get("energy_level")
        if energy_level is not None:
            text = f"Energy Level: {energy_level}"
            cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(img, "Energy Level: UNKNOWN", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # ── 左上: 位置偏差 ──
        deviated = result.get("position_deviation", False)
        offset_x = result.get("offset_x", 0.0)
        offset_y = result.get("offset_y", 0.0)
        dev_text = f"Position: {'DEVIATED' if deviated else 'OK'}  (dx={offset_x:.3f}, dy={offset_y:.3f})"
        dev_color = (0, 0, 255) if deviated else (0, 255, 0)
        cv2.putText(img, dev_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, dev_color, 2)

        # ── 左上: 分隔线 ──
        cv2.line(img, (10, 72), (350, 72), (80, 80, 80), 1)

        # ── 左下: 缺陷类型 ──
        defects = result.get("defects", [])
        base_y = h - 20
        if defects:
            # 有缺陷: 红色背景条 + 缺陷名
            for i, d in enumerate(reversed(defects)):
                y = base_y - i * 28
                defect_name = d["defect_type"]
                conf = d["confidence"]
                text = f"DEFECT: {defect_name} ({conf:.2f})"
                # 红色半透明背景条
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                overlay = img.copy()
                cv2.rectangle(overlay, (8, y - th - 4), (12 + tw, y + 4), (0, 0, 200), -1)
                cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
                cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            text = "Defects: None"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            overlay = img.copy()
            cv2.rectangle(overlay, (8, base_y - th - 4), (12 + tw, base_y + 4), (0, 180, 0), -1)
            cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
            cv2.putText(img, text, (10, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ── 右下: FPS ──
        inf_time = result.get("inference_time_ms", 0)
        fps = 1000.0 / inf_time if inf_time > 0 else 0
        fps_text = f"{fps:.1f} FPS  ({inf_time:.1f}ms)"
        (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(img, fps_text, (w - tw - 10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        return img
