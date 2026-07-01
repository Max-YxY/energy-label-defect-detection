/**
 * @file    ai_detect.c
 * @brief   能效标签缺陷检测 - AI 推理实现
 *
 * 流程: 摄像头帧 → 缩放640x640 → NPU推理 → NMS → 缺陷/偏差判断
 */
#include "ai_detect.h"
#include "network.h"
#include "stai_network.h"
#include <string.h>
#include <math.h>

/* ─── 参数 (对应 config.yaml 最优值) ─── */
#define CONF_THRESHOLD  0.65f
#define IOU_THRESHOLD   0.45f
#define TOLERANCE_X     0.18f
#define TOLERANCE_Y     0.155f
#define EDGE_MARGIN     0.01f
#define AI_INPUT_W      640
#define AI_INPUT_H      640
#define MAX_DETECTIONS  100
#define NUM_CLASSES     10
#define OUTPUT_COLS     8400

/* ─── 全局缓冲 ─── */
static float g_ai_input[3 * AI_INPUT_W * AI_INPUT_H];   /* NCHW */
static float g_ai_output[14 * OUTPUT_COLS];              /* YOLO输出 */
static ai_detection_t g_detections[MAX_DETECTIONS];
static int g_det_count = 0;
static int g_initialized = 0;

/* ─── 工具函数 ─── */
static inline float sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/* NMS */
static void apply_nms(ai_detection_t *dets, int *count, float iou_th) {
    if (*count == 0) return;
    /* 按置信度排序(冒泡) */
    for (int i = 0; i < *count - 1; i++)
        for (int j = i + 1; j < *count; j++)
            if (dets[j].confidence > dets[i].confidence) {
                ai_detection_t t = dets[i]; dets[i] = dets[j]; dets[j] = t;
            }
    int k = 0;
    for (int i = 0; i < *count; i++) {
        int discard = 0;
        for (int j = 0; j < k; j++) {
            int ix1 = dets[i].x1 > dets[j].x1 ? dets[i].x1 : dets[j].x1;
            int iy1 = dets[i].y1 > dets[j].y1 ? dets[i].y1 : dets[j].y1;
            int ix2 = dets[i].x2 < dets[j].x2 ? dets[i].x2 : dets[j].x2;
            int iy2 = dets[i].y2 < dets[j].y2 ? dets[i].y2 : dets[j].y2;
            int iw = ix2 - ix1; if (iw < 0) iw = 0;
            int ih = iy2 - iy1; if (ih < 0) ih = 0;
            int ai = (dets[i].x2 - dets[i].x1) * (dets[i].y2 - dets[i].y1);
            int aj = (dets[j].x2 - dets[j].x1) * (dets[j].y2 - dets[j].y1);
            float iou = (float)(iw * ih) / (float)(ai + aj - iw * ih + 1);
            if (iou > iou_th) { discard = 1; break; }
        }
        if (!discard) dets[k++] = dets[i];
    }
    *count = k;
}

/* ─── YOLO后处理 ─── */
static void parse_output(float *out, ai_detection_t *dets, int *count) {
    *count = 0;
    for (int col = 0; col < OUTPUT_COLS; col++) {
        float cx = out[0 * OUTPUT_COLS + col];
        float cy = out[1 * OUTPUT_COLS + col];
        float w  = out[2 * OUTPUT_COLS + col];
        float h  = out[3 * OUTPUT_COLS + col];
        float best_conf = 0.0f;
        int   best_cls  = -1;
        for (int c = 0; c < NUM_CLASSES; c++) {
            float conf = sigmoid_f(out[(4 + c) * OUTPUT_COLS + col]);
            if (conf > best_conf) { best_conf = conf; best_cls = c; }
        }
        if (best_cls < 0 || best_conf < CONF_THRESHOLD) continue;
        int x1 = (int)(cx - w / 2.0f + 0.5f);
        int y1 = (int)(cy - h / 2.0f + 0.5f);
        int x2 = (int)(cx + w / 2.0f + 0.5f);
        int y2 = (int)(cy + h / 2.0f + 0.5f);
        if (x1 < 0) x1 = 0; if (y1 < 0) y1 = 0;
        if (x2 > AI_INPUT_W-1) x2 = AI_INPUT_W-1;
        if (y2 > AI_INPUT_H-1) y2 = AI_INPUT_H-1;
        dets[*count].class_id = best_cls;
        dets[*count].confidence = best_conf;
        dets[*count].x1 = x1; dets[*count].y1 = y1;
        dets[*count].x2 = x2; dets[*count].y2 = y2;
        (*count)++;
        if (*count >= MAX_DETECTIONS) break;
    }
    apply_nms(dets, count, IOU_THRESHOLD);
}

/* ─── 初始化 ─── */
int ai_detect_init(void)
{
    if (g_initialized) return 0;

    /* 加载 NPU 网络配置并初始化 (由 CubeMX X-CUBE-AI 生成) */
    /* 注意: 此函数由 stedgeai 生成的 stai_network.c 定义 */
    /* 需要在 CubeMX 配置好 X-CUBE-AI 后再生整合 */
    extern bool LL_ATON_EC_Network_Init_network(void);
    LL_ATON_EC_Network_Init_network();

    g_initialized = 1;
    return 0;
}

/* ─── 预处理: RGB888 → float NCHW 640x640 ─── */
static void preprocess(const uint8_t *src, int src_w, int src_h) {
    for (int y = 0; y < AI_INPUT_H; y++) {
        for (int x = 0; x < AI_INPUT_W; x++) {
            int sx = x * src_w / AI_INPUT_W;
            int sy = y * src_h / AI_INPUT_H;
            int idx = (sy * src_w + sx) * 3;
            float r = src[idx]     * (1.0f / 255.0f);
            float g = src[idx + 1] * (1.0f / 255.0f);
            float b = src[idx + 2] * (1.0f / 255.0f);
            g_ai_input[0 * AI_INPUT_H * AI_INPUT_W + y * AI_INPUT_W + x] = r;
            g_ai_input[1 * AI_INPUT_H * AI_INPUT_W + y * AI_INPUT_W + x] = g;
            g_ai_input[2 * AI_INPUT_H * AI_INPUT_W + y * AI_INPUT_W + x] = b;
        }
    }
}

/* ─── 推理 ─── */
int ai_detect_run(const uint8_t *image_rgb888, int w, int h, ai_result_t *result)
{
    if (!g_initialized) return -1;

    /* 1. 拷贝输入到 NPU 输入缓冲区 */
    preprocess(image_rgb888, w, h);
    extern const LL_Buffer_InfoTypeDef *LL_ATON_Input_Buffers_Info_network(void);
    const LL_Buffer_InfoTypeDef *in_info = LL_ATON_Input_Buffers_Info_network();
    memcpy(LL_Buffer_addr_start(in_info), g_ai_input, 3 * AI_INPUT_W * AI_INPUT_H * sizeof(float));
    SCB_InvalidateDCache_by_Addr(LL_Buffer_addr_start(in_info), LL_Buffer_len(in_info));

    /* 2. NPU 推理 */
    extern void LL_ATON_RT_Main_network(void);
    LL_ATON_RT_Main_network();

    /* 从输出缓冲拷贝结果 */
    extern const LL_Buffer_InfoTypeDef *LL_ATON_Output_Buffers_Info_network(void);
    extern unsigned char *LL_Buffer_addr_start(const LL_Buffer_InfoTypeDef *buf);
    const LL_Buffer_InfoTypeDef *out_info = LL_ATON_Output_Buffers_Info_network();
    float *out = (float *)LL_Buffer_addr_start(out_info);
    memcpy(g_ai_output, out, 14 * OUTPUT_COLS * sizeof(float));

    /* 3. 后处理 */
    parse_output(g_ai_output, g_detections, &g_det_count);

    /* 4. 解析结果 */
    memset(result, 0, sizeof(*result));
    result->energy_level = -1;

    int box_idx = -1, label_idx = -1;
    for (int i = 0; i < g_det_count; i++) {
        int c = g_detections[i].class_id;
        if (c >= CLS_LEVEL_1 && c <= CLS_LEVEL_5) {
            if (c + 1 > result->energy_level) result->energy_level = c + 1;
        }
        else if (c >= CLS_STAIN && c <= CLS_WRINKLE && result->defect_count < 3) {
            const char *names[] = {"stain", "damage", "wrinkle"};
            strcpy(result->defect_names[result->defect_count], names[c - CLS_STAIN]);
            result->defect_count++;
            result->has_defect = 1;
        }
        else if (c == CLS_LABEL) label_idx = i;
        else if (c == CLS_BOX)   box_idx = i;
    }

    /* 5. 位置偏差判断 */
    if (label_idx >= 0 && box_idx >= 0) {
        ai_detection_t *lbl = &g_detections[label_idx];
        ai_detection_t *box = &g_detections[box_idx];
        float lcx = (lbl->x1 + lbl->x2) / 2.0f;
        float lcy = (lbl->y1 + lbl->y2) / 2.0f;
        float bcx = (box->x1 + box->x2) / 2.0f;
        float bcy = (box->y1 + box->y2) / 2.0f;
        float bw  = (box->x2 - box->x1);
        float bh  = (box->y2 - box->y1);
        if (bw > 0 && bh > 0) {
            result->offset_x = (lcx - bcx) / bw;
            result->offset_y = (lcy - bcy) / bh;
            if (fabsf(result->offset_x) > TOLERANCE_X ||
                fabsf(result->offset_y) > TOLERANCE_Y) {
                result->position_deviation = 1;
            }
        }
    }

    return 0;
}

/* 获取检测框列表 */
int ai_detect_get_boxes(ai_detection_t *boxes, int max_boxes)
{
    int n = g_det_count < max_boxes ? g_det_count : max_boxes;
    memcpy(boxes, g_detections, n * sizeof(ai_detection_t));
    return n;
}
