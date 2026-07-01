#!/usr/bin/env python3
"""
树莓派能效标签缺陷检测 — 完整版
移植自 energy_label_defect_detection/P 项目算法
"""
import cv2
import numpy as np
import onnxruntime as ort
import time
import os
import sys

# ==========================================
# 1. 配置参数（与项目 config.yaml 一致）
# ==========================================
CLASS_NAMES = {0: 'level_1', 1: 'level_2', 2: 'level_3', 3: 'level_4', 4: 'level_5',
               5: 'stain', 6: 'damage', 7: 'wrinkle', 8: 'label', 9: 'box'}
NC = 10
LABEL_ID = 8
BOX_ID = 9
ENERGY_LEVEL_IDS = {0, 1, 2, 3, 4}
DEFECT_IDS = {5, 6, 7}

# 模型参数
MAIN_INPUT_SIZE = 320          # best.onnx 输入尺寸（平衡精度与速度）
BOX_INPUT_SIZE = 320           # box_detector.onnx 输入尺寸

# 推理参数
CONF_THRESHOLD = 0.25          # 主模型置信度（416模型降低一点）
IOU_THRESHOLD = 0.45
BOX_DETECTOR_CONF = 0.08       # Box detector 低阈值
CV_FALLBACK_CONF = 0.35        # CV 降级置信度

# 位置偏差参数
TOLERANCE_X = 0.18
TOLERANCE_Y = 0.155
EDGE_MARGIN = 0.01
POS_CONF_TH = 0.3

# ==========================================
# 2. 加载 ONNX 模型
# ==========================================
MODEL_DIR = "/home/pi/yolo_model"
main_path = os.path.join(MODEL_DIR, "best.onnx")
box_path = os.path.join(MODEL_DIR, "box_detector.onnx")

if not os.path.exists(main_path):
    print(f"错误: 找不到模型 {main_path}")
    sys.exit(1)

# 主模型
so = ort.SessionOptions()
so.enable_cpu_mem_arena = True
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(main_path, sess_options=so, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
print(f"主模型加载成功: {main_path}")
print(f"  输入: {session.get_inputs()[0].shape}")
print(f"  输出: {session.get_outputs()[0].shape}")

# Box detector（可选）
box_session = None
if os.path.exists(box_path):
    box_so = ort.SessionOptions()
    box_so.enable_cpu_mem_arena = True
    box_so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    box_session = ort.InferenceSession(box_path, sess_options=box_so, providers=['CPUExecutionProvider'])
    box_input_name = box_session.get_inputs()[0].name
    print(f"Box detector 加载成功: {box_path}")
else:
    print("Box detector 未找到，使用 CV fallback")

# ==========================================
# 3. YOLOv8 后处理函数
# ==========================================
def nms(boxes, scores, iou_threshold):
    """非极大值抑制"""
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


def postprocess(output, img_shape, conf_threshold, iou_threshold):
    """
    YOLOv8 输出后处理
    output: (1, 4+NC, num_preds)
    """
    predictions = np.squeeze(output).T
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


# ==========================================
# 4. CV Fallback: 自适应阈值边缘检测找 box
# ==========================================
def find_box_cv(crop):
    """Locate energy rating box via edge detection on cropped label."""
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


# ==========================================
# 5. 摄像头初始化
# ==========================================
camera_type = None
picam2 = None
cap = None

try:
    from picamera2 import Picamera2
    print("尝试初始化 CSI 摄像头...")
    picam2 = Picamera2()
    # picamera2 默认输出 RGB888，OpenCV 需要 BGR，后面做转换
    # 摄像头采集更高分辨率，模型会缩放到 416x416
    config = picam2.create_preview_configuration(main={"size": (640, 480)})
    picam2.configure(config)
    picam2.start()
    camera_type = "CSI"
    print("CSI 摄像头已启动")
except Exception as e:
    print(f"CSI 摄像头失败 ({e})，尝试 USB...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误: 无法打开摄像头")
        sys.exit(1)
    camera_type = "USB"
    print("USB 摄像头已启动")


# ==========================================
# 6. 两阶段检测
# ==========================================
def detect_label_box(image, boxes, scores, class_ids):
    """
    两阶段: 在 label 区域内用 box detector + CV fallback 找 box
    返回新找到的 box 列表 [(x1, y1, x2, y2, conf)]
    """
    new_boxes = []

    # 检查 stage-1 是否已有 box
    stage1_has_box = BOX_ID in class_ids

    if stage1_has_box:
        return new_boxes  # 一阶段已有 box，跳过两阶段

    # 找 label
    label_indices = [i for i, c in enumerate(class_ids) if c == LABEL_ID]

    for idx in label_indices:
        x1, y1, x2, y2 = boxes[idx]
        if x2 <= x1 or y2 <= y1:
            continue

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        box_found = None

        # Stage 2a: Box detector on crop
        if box_session is not None:
            crop_resized = cv2.resize(crop, (BOX_INPUT_SIZE, BOX_INPUT_SIZE))
            input_tensor = crop_resized.astype(np.float32) / 255.0
            input_tensor = np.transpose(input_tensor, (2, 0, 1))
            input_tensor = np.expand_dims(input_tensor, axis=0)

            box_out = box_session.run(None, {box_input_name: input_tensor})
            # box_detector ONNX 输出 [1, 5, 2100]
            box_pred = np.squeeze(box_out[0]).T  # (2100, 5)
            box_confs = box_pred[:, 4]
            if len(box_confs) > 0:
                best_idx = np.argmax(box_confs)
                if box_confs[best_idx] >= BOX_DETECTOR_CONF:
                    bx, by, bw, bh = box_pred[best_idx, :4]
                    # 缩放回 crop 坐标
                    ch, cw = crop.shape[:2]
                    bx1_c = int((bx - bw/2) * cw)
                    by1_c = int((by - bh/2) * ch)
                    bx2_c = int((bx + bw/2) * cw)
                    by2_c = int((by + bh/2) * ch)
                    box_found = (x1 + bx1_c, y1 + by1_c,
                                 x1 + bx2_c, y1 + by2_c,
                                 float(box_confs[best_idx]))

        # Stage 2b: CV fallback
        if box_found is None:
            rect = find_box_cv(crop)
            if rect is not None:
                bx, by, bw, bh = rect
                box_found = (x1 + bx, y1 + by, x1 + bx + bw, y1 + by + bh, CV_FALLBACK_CONF)

        if box_found is not None:
            new_boxes.append(box_found)

    return new_boxes


# ==========================================
# 7. 位置偏差检测
# ==========================================
def check_position_deviation(all_boxes):
    """检查 label 与 box 之间的位置偏差"""
    label_boxes = [b for b in all_boxes if b['class_id'] == LABEL_ID]
    box_boxes = [b for b in all_boxes if b['class_id'] == BOX_ID]

    if not label_boxes or not box_boxes:
        return False, 0.0, 0.0

    label = label_boxes[0]
    box = box_boxes[0]

    if label['confidence'] < POS_CONF_TH or box['confidence'] < POS_CONF_TH:
        return False, 0.0, 0.0

    lx1, ly1, lx2, ly2 = label['bbox']
    bx1, by1, bx2, by2 = box['bbox']

    label_cx = (lx1 + lx2) / 2.0
    label_cy = (ly1 + ly2) / 2.0
    box_cx = (bx1 + bx2) / 2.0
    box_cy = (by1 + by2) / 2.0
    box_w = bx2 - bx1
    box_h = by2 - by1

    if box_w <= 0 or box_h <= 0:
        return False, 0.0, 0.0

    offset_x = (label_cx - box_cx) / box_w
    offset_y = (label_cy - box_cy) / box_h

    deviated = False
    if abs(offset_x) > TOLERANCE_X or abs(offset_y) > TOLERANCE_Y:
        deviated = True
    else:
        left = abs((lx1 - bx1) / box_w)
        right = abs((bx2 - lx2) / box_w)
        top = abs((ly1 - by1) / box_h)
        bottom = abs((by2 - ly2) / box_h)
        if min(left, right, top, bottom) < EDGE_MARGIN:
            deviated = True

    return deviated, offset_x, offset_y


# ==========================================
# 8. 绘制函数（V2 风格）
# ==========================================
def draw_v2(image, all_boxes, energy_level, defects, deviated, offset_x, offset_y, infer_time_ms):
    """V2 绘制布局"""
    img = image.copy()
    h, w = img.shape[:2]

    # 颜色
    colors = {
        'level': (0, 255, 0),
        'defect': (0, 0, 255),
        'label': (255, 255, 0),
        'box': (255, 0, 0),
    }

    # 绘制检测框
    for box in all_boxes:
        cls_id = box['class_id']
        x1, y1, x2, y2 = [int(v) for v in box['bbox']]
        name = box['class_name']
        conf = box['confidence']

        if cls_id in ENERGY_LEVEL_IDS:
            color = colors['level']
        elif cls_id in DEFECT_IDS:
            color = colors['defect']
        elif cls_id == LABEL_ID:
            color = colors['label']
        elif cls_id == BOX_ID:
            color = colors['box']
        else:
            color = (128, 128, 128)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label_text = f"{name} {conf:.2f}"
        cv2.putText(img, label_text, (x1, max(y1 - 6, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # 左上: Energy Level
    if energy_level is not None:
        text = f"Level: {energy_level}"
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(img, "Level: N/A", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 左上: Position
    dev_text = f"Position: {'DEV' if deviated else 'OK'}  (dx={offset_x:.3f}, dy={offset_y:.3f})"
    dev_color = (0, 0, 255) if deviated else (0, 255, 0)
    cv2.putText(img, dev_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, dev_color, 2)

    # 分隔线
    cv2.line(img, (10, 65), (350, 65), (80, 80, 80), 1)

    # 左下: Defects
    base_y = h - 15
    if defects:
        for i, d in enumerate(reversed(defects)):
            y = base_y - i * 24
            text = f"{d['defect_type']} ({d['confidence']:.2f})"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            overlay = img.copy()
            cv2.rectangle(overlay, (8, y - th - 4), (12 + tw, y + 4), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
            cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    else:
        text = "No Defects"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        overlay = img.copy()
        cv2.rectangle(overlay, (8, base_y - th - 4), (12 + tw, base_y + 4), (0, 180, 0), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
        cv2.putText(img, text, (10, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # 右下: FPS
    fps = 1000.0 / infer_time_ms if infer_time_ms > 0 else 0
    fps_text = f"{fps:.1f} FPS  ({infer_time_ms:.1f}ms)"
    (tw, th), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(img, fps_text, (w - tw - 10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return img


# ==========================================
# 9. 主循环
# ==========================================
print("\n开始实时检测，按 Q 退出")
print("=" * 50)

frame_count = 0
try:
    while True:
        # 获取帧
        t_start = time.perf_counter()

        if camera_type == "CSI":
            frame = picam2.capture_array("main")
            # picamera2 输出 RGB → 转 BGR 供 OpenCV 显示/保存
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = cap.read()
            if not ret:
                print("警告: 摄像头读取失败")
                break

        orig_h, orig_w = frame.shape[:2]

        # ---- 预处理 ----
        input_img = cv2.resize(frame, (MAIN_INPUT_SIZE, MAIN_INPUT_SIZE))
        input_tensor = input_img.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)

        # ---- Stage 1: 主模型推理 ----
        outputs = session.run(None, {input_name: input_tensor})
        infer_start = time.perf_counter()
        # 修正: 实际推理时间在 session.run 之后
        infer_time = (time.perf_counter() - t_start) * 1000

        # ---- 后处理 ----
        output = outputs[0]
        boxes, scores, class_ids = postprocess(output, (orig_h, orig_w),
                                                CONF_THRESHOLD, IOU_THRESHOLD)

        # ---- 组装检测结果 ----
        all_boxes = []
        for box, score, cls_id in zip(boxes, scores, class_ids):
            all_boxes.append({
                'class_id': int(cls_id),
                'class_name': CLASS_NAMES.get(int(cls_id), 'unknown'),
                'confidence': float(score),
                'bbox': [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
            })

        # ---- Stage 2: 两阶段 box 检测 ----
        if len(boxes) > 0:
            new_boxes = detect_label_box(frame, boxes, scores, class_ids)
            for nb in new_boxes:
                all_boxes.append({
                    'class_id': BOX_ID,
                    'class_name': 'box',
                    'confidence': nb[4],
                    'bbox': [float(nb[0]), float(nb[1]), float(nb[2]), float(nb[3])],
                })

        # ---- 提取能效等级 ----
        energy_level = None
        energy_boxes = [b for b in all_boxes if b['class_id'] in ENERGY_LEVEL_IDS]
        if energy_boxes:
            energy_boxes.sort(key=lambda b: b['confidence'], reverse=True)
            energy_level = energy_boxes[0]['class_id'] + 1  # class 0 → level 1

        # ---- 提取缺陷 ----
        seen_classes = set()
        defects = []
        defect_boxes = [b for b in all_boxes if b['class_id'] in DEFECT_IDS]
        defect_boxes.sort(key=lambda b: b['confidence'], reverse=True)
        for b in defect_boxes:
            if b['class_id'] not in seen_classes:
                defects.append({
                    'defect_type': b['class_name'],
                    'confidence': b['confidence'],
                })
                seen_classes.add(b['class_id'])

        # ---- 位置偏差 ----
        deviated, offset_x, offset_y = check_position_deviation(all_boxes)

        # ---- 绘制 ----
        annotated = draw_v2(frame, all_boxes, energy_level, defects,
                            deviated, offset_x, offset_y, infer_time)

        # ---- 保存 ----
        cv2.imwrite("/tmp/detected_frame.jpg", annotated)

        # ---- 在右上角画 label 裁剪画中画 ----
        label_crops = [b for b in all_boxes if b['class_id'] == LABEL_ID]
        pip_w, pip_h = 160, 128  # 画中画尺寸
        if label_crops:
            lb = label_crops[0]
            lx1, ly1, lx2, ly2 = [int(v) for v in lb['bbox']]
            mw = int((lx2 - lx1) * 0.05)
            mh = int((ly2 - ly1) * 0.05)
            lx1 = max(0, lx1 - mw)
            ly1 = max(0, ly1 - mh)
            lx2 = min(frame.shape[1], lx2 + mw)
            ly2 = min(frame.shape[0], ly2 + mh)
            if lx2 > lx1 and ly2 > ly1:
                crop = annotated[ly1:ly2, lx1:lx2]  # 从已标注的画面裁剪
                pip = cv2.resize(crop, (pip_w, pip_h))
                # 画白色边框
                cv2.rectangle(pip, (0, 0), (pip_w-1, pip_h-1), (255, 255, 255), 2)
                # 标 "CROP" 文字
                cv2.putText(pip, "CROP", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                # 覆盖到右上角
                annotated[5:5+pip_h, annotated.shape[1]-pip_w-5:annotated.shape[1]-5] = pip
                # 画外框
                cv2.rectangle(annotated,
                    (annotated.shape[1]-pip_w-6, 4),
                    (annotated.shape[1]-4, 5+pip_h),
                    (255, 255, 255), 1)

        # ---- 显示（全屏） ----
        cv2.namedWindow("Energy Label Defect Detection", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Energy Label Defect Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Energy Label Defect Detection", annotated)

        # ---- 日志 ----
        frame_count += 1
        log_parts = [f"[#{frame_count} {infer_time:.0f}ms]"]
        if energy_level:
            log_parts.append(f"Lv{energy_level}")
        if defects:
            for d in defects:
                log_parts.append(f"{d['defect_type']}({d['confidence']:.2f})")
        log_parts.append(f"Box:{'Y' if BOX_ID in [b['class_id'] for b in all_boxes] else 'N'}")
        log_parts.append(f"Pos:{'DEV' if deviated else 'OK'}")
        print(" ".join(log_parts))

        # ---- 帧率稳定：320x320 推理约 150ms，目标 6 FPS ----
        TARGET_FRAME_MS = 167  # 6 FPS
        elapsed = (time.perf_counter() - t_start) * 1000
        sleep_ms = TARGET_FRAME_MS - elapsed
        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

except KeyboardInterrupt:
    print("\n用户中断")
finally:
    if camera_type == "CSI":
        picam2.stop()
        picam2.close()
    else:
        cap.release()
    cv2.destroyAllWindows()
    print("已退出")
