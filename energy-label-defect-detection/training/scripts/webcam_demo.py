#!/usr/bin/env python3
"""
PC 摄像头实时演示脚本 (优化版)。
- 跳帧推理: 每 3 帧推理一次，其余帧复用上一次结果
- 降低输入分辨率加速
- 平滑显示: 低通滤波减少标签抖动
- 缺陷类型标在左下角, 按 Q 退出
"""
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference.detector import EnergyLabelDetector

CONFIG_PATH = str(Path(__file__).parent.parent / "config.yaml")

# ── 可调参数 ──
SKIP_FRAMES = 2           # 每 N 帧推理一次 (0=每帧都推理)
INPUT_SIZE = (640, 480)   # 摄像头采集分辨率
SMOOTH_ALPHA = 0.3        # 平滑系数 (0=完全不更新, 1=完全信任新值)


def smooth_boxes(old_boxes, new_boxes, alpha=0.3):
    """对检测框做指数平滑，减少抖动."""
    if not old_boxes or not new_boxes:
        return new_boxes
    # 按 class_id + 位置最近匹配
    smoothed = []
    new_matched = set()
    for old in old_boxes:
        best = None
        best_dist = float("inf")
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
            old_b = old["bbox"]
            new_b = new_boxes[best]["bbox"]
            s = {
                **new_boxes[best],
                "bbox": [
                    old_b[0] * (1 - alpha) + new_b[0] * alpha,
                    old_b[1] * (1 - alpha) + new_b[1] * alpha,
                    old_b[2] * (1 - alpha) + new_b[2] * alpha,
                    old_b[3] * (1 - alpha) + new_b[3] * alpha,
                ],
                "confidence": (old["confidence"] + new_boxes[best]["confidence"]) / 2,
            }
            smoothed.append(s)
    # 新增的框直接加入
    for j, new in enumerate(new_boxes):
        if j not in new_matched:
            smoothed.append(new)
    return smoothed


def main():
    print("=" * 50)
    print("  能效标签缺陷检测 — PC 摄像头演示 (优化版)")
    print(f"  跳帧: 每{SKIP_FRAMES+1}帧推理 | 平滑: α={SMOOTH_ALPHA}")
    print("  按 Q 退出")
    print("=" * 50)

    print("\n[1/2] 加载模型...")
    detector = EnergyLabelDetector(CONFIG_PATH)

    print("[2/2] 打开摄像头...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("ERROR: 无法打开摄像头！")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, INPUT_SIZE[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, INPUT_SIZE[1])
    print(f"  分辨率: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}\n")

    frame_idx = 0
    last_result = None
    last_boxes = None
    last_fps_time = time.perf_counter()
    fps_display = 0.0
    real_fps_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            real_fps_count += 1

            # ── 跳帧推理 ──
            if frame_idx % (SKIP_FRAMES + 1) == 0:
                result = detector.detect(frame)
                # 平滑检测框
                if last_boxes is not None:
                    result["boxes"] = smooth_boxes(last_boxes, result["boxes"], SMOOTH_ALPHA)
                last_result = result
                last_boxes = result["boxes"]
            else:
                result = last_result

            frame_idx += 1

            # ── 绘制 ──
            if result is not None:
                annotated = detector.postprocessor.draw_v2(frame, result)
            else:
                annotated = frame.copy()

            # FPS 计算
            t_now = time.perf_counter()
            if t_now - last_fps_time >= 1.0:
                fps_display = real_fps_count / (t_now - last_fps_time)
                real_fps_count = 0
                last_fps_time = t_now

            cv2.putText(annotated, f"Real: {fps_display:.1f} FPS",
                        (annotated.shape[1] - 170, annotated.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow("Energy Label Defect Detection", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                print("\n用户退出。")
                break

    except KeyboardInterrupt:
        print("\n中断。")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("已释放摄像头。")


if __name__ == "__main__":
    main()
