#ifndef __POSTPROCESS_H
#define __POSTPROCESS_H

#include <stdint.h>

/* 检测结果结构体 */
typedef struct {
    int class_id;           /* 类别ID 0-9 */
    float confidence;       /* 置信度 */
    int x1, y1, x2, y2;    /* 边界框 (像素坐标) */
} Detection;

typedef struct {
    int energy_level;       /* 能效等级 1-5, 0=未知 */
    int defect_count;       /* 缺陷数量 */
    char defect_names[3][16]; /* 缺陷名称 */
    int has_stain;          /* 是否有污渍 */
    int has_damage;         /* 是否有破损 */
    int has_wrinkle;        /* 是否有褶皱 */
    int pos_deviation;      /* 位置是否偏移 */
    int detection_count;    /* 总检测框数 */
    Detection detections[32]; /* 检测框列表 */
} DetectionResult;

/* YOLOv8 NPU 输出后处理 */
void postprocess_yolov8(
    const float *npu_output,    /* NPU输出: [14][8400] float32 */
    int img_width,              /* 原始图像宽度 */
    int img_height,             /* 原始图像高度 */
    float conf_threshold,       /* 置信度阈值 (如 0.65) */
    float iou_threshold,        /* IoU阈值 (如 0.45) */
    DetectionResult *result     /* 输出结果 */
);

#endif /* __POSTPROCESS_H */
