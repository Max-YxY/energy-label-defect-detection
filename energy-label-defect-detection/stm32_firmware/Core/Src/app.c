#include "app.h"
#include "app_config.h"
#include "app_lcd.h"
#include "app_camera.h"
#include "postprocess.h"
#include "ll_aton_runtime.h"
#include "stm32_lcd.h"
#include "stm32_lcd_ex.h"
#include <stdio.h>
#include <string.h>

/* NPU 输入输出缓冲区 */
static uint8_t *g_nn_input = NULL;
static float_t *g_nn_output[NN_OUTPUT_NUMBER];
static uint32_t g_nn_output_len[NN_OUTPUT_NUMBER];
static volatile uint32_t g_frame_ready = 0;
static uint32_t g_frame_count = 0;

/* 回调 */
static void app_camera_nn_pipe_frame_cb(void);

/* 检测框颜色 */
#define COLOR_GREEN     0x07E0
#define COLOR_RED       0xF800
#define COLOR_YELLOW    0xFFE0
#define COLOR_BLUE      0x001F
#define COLOR_WHITE     0xFFFF
#define COLOR_BLACK     0x0000

/**
 * @brief  在 LCD 前景层绘制检测结果
 */
static void app_draw_detections(DetectionResult *result, uint32_t inference_ms)
{
    char info[64];
    int line = 0;

    /* 清空前景层 */
    UTIL_LCD_FillRect(0, 0, LCD_FG_WIDTH, LCD_FG_HEIGHT, COLOR_BLACK);

    /* ── 左上角信息 ── */
    /* 帧率/推理时间 */
    UTIL_LCDEx_PrintfAt(10, LINE(line), RIGHT_MODE, "%dms  %dFPS",
        inference_ms, g_frame_count * 1000 / (HAL_GetTick() + 1));
    line += 2;

    /* 能效等级 */
    if (result->energy_level > 0) {
        snprintf(info, sizeof(info), "Level: %d", result->energy_level);
        UTIL_LCDEx_PrintfAt(10, LINE(line), RIGHT_MODE, "%s", info);
    } else {
        UTIL_LCDEx_PrintfAt(10, LINE(line), RIGHT_MODE, "Level: ?");
    }
    line += 2;

    /* 缺陷 */
    if (result->defect_count > 0) {
        for (int i = 0; i < result->defect_count; i++) {
            UTIL_LCDEx_PrintfAt(10, LINE(line), RIGHT_MODE, "%s",
                result->defect_names[i]);
            UTIL_LCD_FillRect(8, LINE(line) - 12, 200, 16, COLOR_RED);
            line += 1;
        }
    } else {
        UTIL_LCDEx_PrintfAt(10, LINE(line), RIGHT_MODE, "No Defects");
        line += 1;
    }

    /* 位置 */
    if (result->pos_deviation) {
        UTIL_LCDEx_PrintfAt(10, LINE(line), RIGHT_MODE, "POS DEVIATED");
    }

    /* ── 绘制检测框 ── */
    for (int i = 0; i < result->detection_count; i++) {
        Detection *det = &result->detections[i];
        uint32_t color;
        int cid = det->class_id;

        /* 选颜色 */
        if (cid >= 0 && cid <= 4)       color = COLOR_GREEN;
        else if (cid >= 5 && cid <= 7)  color = COLOR_RED;
        else if (cid == 8)              color = COLOR_YELLOW;
        else if (cid == 9)              color = COLOR_BLUE;
        else                            color = COLOR_WHITE;

        /* 画框（用前景层绘制矩形） */
        UTIL_LCD_DrawRect(det->x1, det->y1,
                          det->x2 - det->x1, det->y2 - det->y1, color);

        /* 画标签 */
        snprintf(info, sizeof(info), "%s %.0f%%",
                 NN_CLASSES_TABLE[det->class_id],
                 det->confidence * 100);
        UTIL_LCDEx_PrintfAt(det->x1 + 2, det->y1 + 2, RIGHT_MODE, "%s", info);
    }

    /* 提交前景层到显示 */
    app_lcd_draw_area_commit();
}

void app_run(void)
{
    /* ── 获取 NPU 模型信息 ── */
    LL_ATON_DECLARE_NAMED_NN_INSTANCE_AND_INTERFACE(Default);
    const LL_Buffer_InfoTypeDef *nn_in_info = LL_ATON_Input_Buffers_Info_Default();
    const LL_Buffer_InfoTypeDef *nn_out_info = LL_ATON_Output_Buffers_Info_Default();

    g_nn_input = (uint8_t *)LL_Buffer_addr_start(&nn_in_info[0]);
    for (int i = 0; i < NN_OUTPUT_NUMBER; i++)
    {
        g_nn_output[i] = (float_t *)LL_Buffer_addr_start(&nn_out_info[i]);
        g_nn_output_len[i] = LL_Buffer_len(&nn_out_info[i]);
    }

    /* ── 初始化 LCD ── */
    app_lcd_init();
    app_lcd_draw_area_update();
    UTIL_LCD_FillRect(0, 0, LCD_FG_WIDTH, LCD_FG_HEIGHT, COLOR_BLACK);
    UTIL_LCDEx_PrintfAt(10, 50, RIGHT_MODE, "Energy Label Detection");
    UTIL_LCDEx_PrintfAt(10, 80, RIGHT_MODE, "YOLOv8n on NPU");
    app_lcd_draw_area_commit();

    /* ── 初始化摄像头（自动检测 OV5640 / IMX335）── */
    app_camera_init(NULL, NULL, NULL, app_camera_nn_pipe_frame_cb);

    /* ── 启动实时预览 → LCD ── */
    app_camera_display_pipe_start(app_lcd_get_bg_buffer(), CMW_MODE_CONTINUOUS);

    /* ── 主循环 ── */
    while (1)
    {
        app_camera_isp_update();

        /* 触发 NPU 管道采集 */
        app_camera_nn_pipe_start(g_nn_input, CMW_MODE_SNAPSHOT);
        while (g_frame_ready == 0);
        g_frame_ready = 0;

        /* 刷 D-Cache */
        SCB_CleanInvalidateDCache();

        /* NPU 推理 */
        uint32_t t0 = HAL_GetTick();
        LL_ATON_RT_Main(&NN_Instance_Default);
        uint32_t inference_ms = HAL_GetTick() - t0;

        /* 读 NPU 输出 */
        for (int i = 0; i < NN_OUTPUT_NUMBER; i++) {
            SCB_InvalidateDCache_by_Addr(g_nn_output[i], g_nn_output_len[i]);
        }

        /* YOLOv8 后处理 */
        DetectionResult result;
        postprocess_yolov8(
            (const float *)g_nn_output[0],
            LCD_FG_WIDTH,
            LCD_FG_HEIGHT,
            NN_CONF_THRESHOLD,
            NN_IOU_THRESHOLD,
            &result);

        /* 绘制检测结果 */
        app_draw_detections(&result, inference_ms);

        g_frame_count++;
    }
}

static void app_camera_nn_pipe_frame_cb(void)
{
    g_frame_ready = 1;
}
