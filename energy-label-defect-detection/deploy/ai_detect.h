/**
 * @file    ai_detect.h
 * @brief   能效标签缺陷检测 - AI 推理模块头文件
 */

#ifndef AI_DETECT_H
#define AI_DETECT_H

#include <stdint.h>

/* 检测类别 */
enum {
    CLS_LEVEL_1=0, CLS_LEVEL_2, CLS_LEVEL_3, CLS_LEVEL_4, CLS_LEVEL_5,
    CLS_STAIN=5, CLS_DAMAGE=6, CLS_WRINKLE=7,
    CLS_LABEL=8, CLS_BOX=9
};

/* 检测结果 */
typedef struct {
    uint16_t class_id;
    float    confidence;
    int16_t  x1, y1, x2, y2;  /* 640x640 坐标 */
} ai_detection_t;

/* 缺陷信息 */
typedef struct {
    int   energy_level;     /* 1-5, -1 表示未检测到 */
    int   has_defect;
    char  defect_names[3][16]; /* stain/damage/wrinkle */
    int   defect_count;
    int   position_deviation;
    float offset_x, offset_y;
} ai_result_t;

/* 初始化 AI 模块 */
int ai_detect_init(void);

/* 运行一次推理 (image: RGB888, 640x480) */
/* result: 输出检测结果 */
int ai_detect_run(const uint8_t *image_rgb888, int w, int h, ai_result_t *result);

/* 获取检测到的目标列表 */
int ai_detect_get_boxes(ai_detection_t *boxes, int max_boxes);

#endif
