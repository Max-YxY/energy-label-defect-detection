#ifndef __APP_CONFIG_H
#define __APP_CONFIG_H

#include "stm32n6xx_hal.h"

/* LCD 配置 */
#define LCD_BG_WIDTH                    480
#define LCD_BG_HEIGHT                   480
#define LCD_FG_WIDTH                    800
#define LCD_FG_HEIGHT                   480

/* 摄像头配置（OV5640 或 IMX335） */
#define CAMERA_MIRROR_FLIP              CMW_MIRRORFLIP_MIRROR

/* YOLOv8 NPU 模型配置 */
#define NN_WIDTH                        320     /* 模型输入宽度 */
#define NN_HEIGHT                       320     /* 模型输入高度 */
#define NN_FORMAT                       DCMIPP_PIXEL_PACKER_FORMAT_RGB888_YUV444_1
#define NN_BPP                          3

/* YOLOv8 检测参数 */
#define NN_CONF_THRESHOLD               0.55f   /* 置信度阈值 */
#define NN_IOU_THRESHOLD                0.45f   /* IoU 阈值 */
#define NN_ANCHORS                      8400    /* 锚点数 */
#define NN_CLASSES                      10      /* 类别数 */
#define NN_OUTPUT_NUMBER                1       /* 输出张量数 */

/* 类别名称 */
#define NN_CLASSES_TABLE                {\
                                            "level_1",\
                                            "level_2",\
                                            "level_3",\
                                            "level_4",\
                                            "level_5",\
                                            "stain",\
                                            "damage",\
                                            "wrinkle",\
                                            "label",\
                                            "box"\
                                        }

/* 类别分组 */
#define ENERGY_LEVEL_IDS                {0, 1, 2, 3, 4}
#define DEFECT_IDS                      {5, 6, 7}
#define LABEL_ID                        8
#define BOX_ID                          9

#endif /* __APP_CONFIG_H */
