#include "postprocess.h"
#include <string.h>
#include <math.h>

/* 类别名称 */
static const char *CLASS_NAMES[10] = {
    "level_1", "level_2", "level_3", "level_4", "level_5",
    "stain", "damage", "wrinkle", "label", "box"
};

/* 类别分组 */
#define ENERGY_LEVEL_IDS_START 0
#define ENERGY_LEVEL_IDS_END   4
#define DEFECT_IDS_START       5
#define DEFECT_IDS_END         7
#define LABEL_ID               8
#define BOX_ID                 9

/* 计算两个框的IoU */
static float compute_iou(int ax1, int ay1, int ax2, int ay2,
                         int bx1, int by1, int bx2, int by2)
{
    int inter_x1 = ax1 > bx1 ? ax1 : bx1;
    int inter_y1 = ay1 > by1 ? ay1 : by1;
    int inter_x2 = ax2 < bx2 ? ax2 : bx2;
    int inter_y2 = ay2 < by2 ? ay2 : by2;

    int inter_w = inter_x2 - inter_x1;
    int inter_h = inter_y2 - inter_y1;
    if (inter_w < 0) inter_w = 0;
    if (inter_h < 0) inter_h = 0;
    float inter_area = (float)(inter_w * inter_h);

    float area_a = (float)((ax2 - ax1) * (ay2 - ay1));
    float area_b = (float)((bx2 - bx1) * (by2 - by1));
    float union_area = area_a + area_b - inter_area;

    if (union_area < 1e-6f) return 0.0f;
    return inter_area / union_area;
}

/* NMS 非极大值抑制 */
static int nms_filter(Detection *dets, int count, float iou_threshold)
{
    int kept[32] = {0};
    int kept_count = 0;

    /* 按置信度降序排列（简单选择排序） */
    int order[32];
    for (int i = 0; i < count; i++) order[i] = i;

    for (int i = 0; i < count - 1; i++) {
        int best = i;
        for (int j = i + 1; j < count; j++) {
            if (dets[order[j]].confidence > dets[order[best]].confidence)
                best = j;
        }
        int tmp = order[i];
        order[i] = order[best];
        order[best] = tmp;
    }

    for (int i = 0; i < count; i++) {
        int keep = 1;
        for (int j = 0; j < kept_count; j++) {
            float iou = compute_iou(
                dets[order[i]].x1, dets[order[i]].y1,
                dets[order[i]].x2, dets[order[i]].y2,
                dets[kept[j]].x1, dets[kept[j]].y1,
                dets[kept[j]].x2, dets[kept[j]].y2);
            if (iou > iou_threshold) {
                keep = 0;
                break;
            }
        }
        if (keep) {
            kept[kept_count++] = order[i];
        }
    }

    /* 重排结果 */
    for (int i = 0; i < kept_count; i++) {
        if (kept[i] != i) {
            dets[i] = dets[kept[i]];
        }
    }
    return kept_count;
}

void postprocess_yolov8(
    const float *npu_output,
    int img_width,
    int img_height,
    float conf_threshold,
    float iou_threshold,
    DetectionResult *result)
{
    int i, j;

    /* 清空结果 */
    memset(result, 0, sizeof(DetectionResult));

    /* NPU输出格式: [14][8400], CHW */
    /* 每列: [cx, cy, w, h, cls0, cls1, ..., cls9] */
    /* 坐标相对于640归一化到[0,1], 类别分数已sigmoid */

    Detection temp_dets[32];
    int det_count = 0;

    for (int anchor = 0; anchor < 8400 && det_count < 32; anchor++) {
        /* 取类别分数最大值 */
        float max_score = 0.0f;
        int max_cls = 0;
        for (int c = 0; c < 10; c++) {
            float score = npu_output[(4 + c) * 8400 + anchor];
            if (score > max_score) {
                max_score = score;
                max_cls = c;
            }
        }

        if (max_score < conf_threshold) continue;

        /* 取bbox (归一化到[0,1]) */
        float cx = npu_output[0 * 8400 + anchor];
        float cy = npu_output[1 * 8400 + anchor];
        float w  = npu_output[2 * 8400 + anchor];
        float h  = npu_output[3 * 8400 + anchor];

        /* 转成像素坐标 */
        int x1 = (int)((cx - w / 2.0f) * img_width);
        int y1 = (int)((cy - h / 2.0f) * img_height);
        int x2 = (int)((cx + w / 2.0f) * img_width);
        int y2 = (int)((cy + h / 2.0f) * img_height);

        /* 裁剪到图像范围内 */
        if (x1 < 0) x1 = 0;
        if (y1 < 0) y1 = 0;
        if (x2 > img_width - 1) x2 = img_width - 1;
        if (y2 > img_height - 1) y2 = img_height - 1;
        if (x2 <= x1 || y2 <= y1) continue;

        temp_dets[det_count].class_id = max_cls;
        temp_dets[det_count].confidence = max_score;
        temp_dets[det_count].x1 = x1;
        temp_dets[det_count].y1 = y1;
        temp_dets[det_count].x2 = x2;
        temp_dets[det_count].y2 = y2;
        det_count++;
    }

    /* NMS */
    det_count = nms_filter(temp_dets, det_count, iou_threshold);

    /* 复制到结果 */
    result->detection_count = det_count;
    for (i = 0; i < det_count && i < 32; i++) {
        result->detections[i] = temp_dets[i];
    }

    /* ===== 提取能效等级 ===== */
    float best_energy_conf = 0.0f;
    for (i = 0; i < det_count; i++) {
        int cid = temp_dets[i].class_id;
        if (cid >= ENERGY_LEVEL_IDS_START && cid <= ENERGY_LEVEL_IDS_END) {
            if (temp_dets[i].confidence > best_energy_conf) {
                best_energy_conf = temp_dets[i].confidence;
                result->energy_level = cid + 1; /* class 0 → level 1 */
            }
        }
    }

    /* ===== 提取缺陷（去重） ===== */
    result->defect_count = 0;
    for (i = 0; i < det_count; i++) {
        int cid = temp_dets[i].class_id;
        if (cid >= DEFECT_IDS_START && cid <= DEFECT_IDS_END) {
            /* 检查是否已有同类缺陷 */
            int already = 0;
            for (j = 0; j < result->defect_count; j++) {
                if (strcmp(result->defect_names[j], CLASS_NAMES[cid]) == 0) {
                    already = 1;
                    break;
                }
            }
            if (!already && result->defect_count < 3) {
                strncpy(result->defect_names[result->defect_count],
                        CLASS_NAMES[cid], 15);
                result->defect_count++;
                if (cid == 5) result->has_stain = 1;
                if (cid == 6) result->has_damage = 1;
                if (cid == 7) result->has_wrinkle = 1;
            }
        }
    }

    /* ===== 位置偏差（需要 label + box 同时存在） ===== */
    Detection *label_det = NULL;
    Detection *box_det = NULL;
    for (i = 0; i < det_count; i++) {
        if (temp_dets[i].class_id == LABEL_ID) label_det = &temp_dets[i];
        if (temp_dets[i].class_id == BOX_ID) box_det = &temp_dets[i];
    }

    if (label_det && box_det && label_det->confidence >= 0.3f && box_det->confidence >= 0.3f) {
        float lcx = (float)(label_det->x1 + label_det->x2) / 2.0f;
        float lcy = (float)(label_det->y1 + label_det->y2) / 2.0f;
        float bcx = (float)(box_det->x1 + box_det->x2) / 2.0f;
        float bcy = (float)(box_det->y1 + box_det->y2) / 2.0f;
        float bw = (float)(box_det->x2 - box_det->x1);
        float bh = (float)(box_det->y2 - box_det->y1);

        if (bw > 0 && bh > 0) {
            float ox = (lcx - bcx) / bw;
            float oy = (lcy - bcy) / bh;

            if (ox < -0.18f) ox = -0.18f;
            if (ox > 0.18f) ox = 0.18f;
            if (oy < -0.155f) oy = -0.155f;
            if (oy > 0.155f) oy = 0.155f;

            if (fabsf(ox) > 0.18f || fabsf(oy) > 0.155f) {
                result->pos_deviation = 1;
            }
        }
    }
}
