/**
  ******************************************************************************
  * @file    app_energy_label.c
  * @brief   能效标签缺陷检测 — STM32N647 应用代码
  *
  * 硬件: STM32N647 + OV5640 + 4.3"RGB 800×480 + TF卡
  * 依赖: STM32 HAL, X-CUBE-AI, OV5640 驱动
  *
  * 在 CubeMX 生成的 main.c USER CODE 中调用:
  *   EnergyLabel_Init();
  *   EnergyLabel_Process();  // while(1) 中循环
  ******************************************************************************
  */

/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "app_x-cube-ai.h"
#include "ov5640_driver.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* ─── 外部声明 (CubeMX 生成的句柄) ─── */
extern DCMI_HandleTypeDef hdcmi;
extern I2C_HandleTypeDef  hi2c;    /* 摄像头控制用 */
extern LTDC_HandleTypeDef hltdc;
extern DMA2D_HandleTypeDef hdma2d;
extern SDRAM_HandleTypeDef hsdram;

extern void MX_DCIMI_Init(void);
extern void MX_LTDC_Init(void);
extern void MX_FMC_Init(void);
extern void MX_I2C1_Init(void);
extern void MX_DMA2D_Init(void);

/* ─── 配置参数 (对应 config.yaml 最优参数) ─── */
#define CONF_THRESHOLD          0.65f
#define IOU_THRESHOLD           0.45f
#define TOLERANCE_X             0.18f
#define TOLERANCE_Y             0.155f
#define EDGE_MARGIN             0.01f

/* ─── 类别 ID ─── */
enum {
    CLASS_LEVEL_1, CLASS_LEVEL_2, CLASS_LEVEL_3, CLASS_LEVEL_4, CLASS_LEVEL_5,
    CLASS_STAIN, CLASS_DAMAGE, CLASS_WRINKLE,
    CLASS_LABEL, CLASS_BOX,
    NUM_CLASSES
};

/* ─── 推理输入/输出尺寸 ─── */
#define AI_INPUT_W    640
#define AI_INPUT_H    640
#define LCD_W         800
#define LCD_H         480
#define CAM_W         640   /* OV5640 输出 VGA */
#define CAM_H         480
#define FRAME_BUFFER_SIZE  (LCD_W * LCD_H * 2)  /* RGB565 */

/* ─── LCD 帧缓冲地址 (SDRAM) ─── */
#define LCD_FRAME_BUF_ADDR  0x70000000  /* 根据 SDRAM 映射调整 */
#define CAM_FRAME_BUF_ADDR  (LCD_FRAME_BUF_ADDR + FRAME_BUFFER_SIZE)
#define AI_INPUT_BUF_ADDR   (CAM_FRAME_BUF_ADDR + CAM_W * CAM_H * 2)

/* ─── 检测结果 ─── */
typedef struct {
    uint16_t class_id;
    float    confidence;
    int16_t  x1, y1, x2, y2;
} Detection_t;
#define MAX_DETS 100

/* ─── 颜色 (RGB565) ─── */
#define CL_BLACK    0x0000
#define CL_GREEN    0x07E0
#define CL_RED      0xF800
#define CL_YELLOW   0xFFE0
#define CL_BLUE     0x001F
#define CL_WHITE    0xFFFF
#define CL_GRAY     0x8410
#define CL_DARKRED  0xA800

/* ─── 全局 ─── */
static Detection_t g_dets[MAX_DETS];
static uint16_t g_det_count = 0;
static volatile uint8_t g_frame_ready = 0;
static uint32_t g_frame_cnt = 0;

/* 帧完成中断回调 (DCMI) */
void EnergyLabel_FrameReady(void) {
    g_frame_ready = 1;
}

/* AI buffers (由 CubeMX 生成) */
AI_ALIGNED(4) static float g_ai_in[3 * AI_INPUT_W * AI_INPUT_H];
extern float *g_ai_out;  /* X-CUBE-AI 生成的输出 */

/* ──────── 工具函数 ──────── */

static float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }
static float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* NMS */
static void apply_nms(Detection_t *d, uint16_t *n, float iou_th) {
    if (*n == 0) return;
    for (uint16_t i = 0; i < *n - 1; i++)
        for (uint16_t j = i + 1; j < *n; j++)
            if (d[j].confidence > d[i].confidence) {
                Detection_t t = d[i]; d[i] = d[j]; d[j] = t;
            }
    uint16_t k = 0;
    for (uint16_t i = 0; i < *n; i++) {
        uint8_t disc = 0;
        for (uint16_t j = 0; j < k; j++) {
            int ix1 = d[i].x1 > d[j].x1 ? d[i].x1 : d[j].x1;
            int iy1 = d[i].y1 > d[j].y1 ? d[i].y1 : d[j].y1;
            int ix2 = d[i].x2 < d[j].x2 ? d[i].x2 : d[j].x2;
            int iy2 = d[i].y2 < d[j].y2 ? d[i].y2 : d[j].y2;
            int iw = ix2 - ix1; if (iw < 0) iw = 0;
            int ih = iy2 - iy1; if (ih < 0) ih = 0;
            int ai = (d[i].x2 - d[i].x1) * (d[i].y2 - d[i].y1);
            int aj = (d[j].x2 - d[j].x1) * (d[j].y2 - d[j].y1);
            float iou = (float)(iw * ih) / (float)(ai + aj - iw * ih);
            if (iou > iou_th) { disc = 1; break; }
        }
        if (!disc) d[k++] = d[i];
    }
    *n = k;
}

/* 坐标: 640×640 → LCD 800×480 */
static void map_pt(int *x, int *y) {
    *x = *x * LCD_W / AI_INPUT_W;
    *y = *y * LCD_H / AI_INPUT_H;
}

/* ──────── YOLOv8 后处理 ──────── */
static void parse_output(float *out, Detection_t *d, uint16_t *n) {
    *n = 0;
    /* YOLOv8 输出: (1, 14, 8400) */
    for (int col = 0; col < 8400; col++) {
        float cx = out[0 * 8400 + col];
        float cy = out[1 * 8400 + col];
        float w  = out[2 * 8400 + col];
        float h  = out[3 * 8400 + col];
        float best_conf = 0;
        int   best_cls  = -1;
        for (int c = 0; c < NUM_CLASSES; c++) {
            float conf = sigmoidf(out[(4 + c) * 8400 + col]);
            if (conf > best_conf) { best_conf = conf; best_cls = c; }
        }
        if (best_cls < 0 || best_conf < CONF_THRESHOLD) continue;
        int x1 = (int)(cx - w / 2 + 0.5f);
        int y1 = (int)(cy - h / 2 + 0.5f);
        int x2 = (int)(cx + w / 2 + 0.5f);
        int y2 = (int)(cy + h / 2 + 0.5f);
        if (x1 < 0) x1 = 0; if (y1 < 0) y1 = 0;
        if (x2 > AI_INPUT_W - 1) x2 = AI_INPUT_W - 1;
        if (y2 > AI_INPUT_H - 1) y2 = AI_INPUT_H - 1;
        d[*n].class_id = best_cls;
        d[*n].confidence = best_conf;
        d[*n].x1 = x1; d[*n].y1 = y1;
        d[*n].x2 = x2; d[*n].y2 = y2;
        (*n)++;
        if (*n >= MAX_DETS) break;
    }
    apply_nms(d, n, IOU_THRESHOLD);
}

/* ──────── LCD 绘制 ──────── */

static void lcd_set_pixel(int x, int y, uint16_t cl) {
    if (x < 0 || x >= LCD_W || y < 0 || y >= LCD_H) return;
    *(volatile uint16_t *)(LCD_FRAME_BUF_ADDR + (y * LCD_W + x) * 2) = cl;
}

static void lcd_draw_rect(int x1, int y1, int x2, int y2, uint16_t cl) {
    map_pt(&x1, &y1); map_pt(&x2, &y2);
    for (int x = x1; x <= x2; x++) { lcd_set_pixel(x, y1, cl); lcd_set_pixel(x, y2, cl); }
    for (int y = y1; y <= y2; y++) { lcd_set_pixel(x1, y, cl); lcd_set_pixel(x2, y, cl); }
}

static void lcd_fill_rect(int x1, int y1, int x2, int y2, uint16_t cl) {
    if (x1 < 0) x1 = 0; if (y1 < 0) y1 = 0;
    if (x2 >= LCD_W) x2 = LCD_W - 1; if (y2 >= LCD_H) y2 = LCD_H - 1;
    for (int y = y1; y <= y2; y++)
        for (int x = x1; x <= x2; x++)
            lcd_set_pixel(x, y, cl);
}

/* 简易 8x12 字体绘制 */
static void lcd_draw_char(int x, int y, char c, uint16_t fg, uint16_t bg) {
    /* 使用 CubeMX 提供的 Font8 / Font12 等 */
    /* 这里改调用 BSP 或直接 printf 到串口调试 */
    /* 实际部署建议用 CubeMX Fonts: BSP_LCD_DisplayStringAt(x,y,str,CENTER_MODE) */
    (void)x; (void)y; (void)c; (void)fg; (void)bg;
}

static void lcd_draw_str(int x, int y, const char *s, uint16_t fg, uint16_t bg) {
    lcd_fill_rect(x - 2, y - 14, x + 150, y + 2, bg);
    /* 此处调用 LCD 驱动库的字符串函数 */
    /* 例如: BSP_LCD_SetTextColor(fg); BSP_LCD_SetBackColor(bg); */
    /* BSP_LCD_DisplayStringAt(x, y, (uint8_t*)s, LEFT_MODE); */
    (void)x; (void)y; (void)s; (void)fg; (void)bg;
}

/* ──────── 摄像头 ──────── */

/* OV5640 I2C 写回调 */
static void camera_i2c_write(uint8_t dev, uint16_t reg, uint8_t val) {
    uint8_t data[2] = { (reg >> 8) & 0xFF, reg & 0xFF };
    HAL_I2C_Master_Transmit(&hi2c, dev, data, 2, 100);
    data[0] = val;
    HAL_I2C_Master_Transmit(&hi2c, dev, data, 1, 100);
}

/* DCMI 帧完成回调 (在 stm32n6xx_it.c 中) */
void HAL_DCMI_FrameEventCallback(DCMI_HandleTypeDef *hdcmi) {
    g_frame_ready = 1;
}

static void camera_init(void) {
    OV5640_Init(camera_i2c_write);
    HAL_Delay(100);
    /* 启动 DCMI 捕获 */
    HAL_DCMI_Start_DMA(&hdcmi, DCMI_MODE_CONTINUOUS,
                        (uint32_t)CAM_FRAME_BUF_ADDR, CAM_H);
}

/* ──────── RGB565 → YOLO 输入 ──────── */
static void preprocess(void) {
    uint16_t *src = (uint16_t *)CAM_FRAME_BUF_ADDR;
    for (int y = 0; y < AI_INPUT_H; y++) {
        for (int x = 0; x < AI_INPUT_W; x++) {
            int sx = x * CAM_W / AI_INPUT_W;
            int sy = y * CAM_H / AI_INPUT_H;
            uint16_t p = src[sy * CAM_W + sx];
            float r = ((p >> 11) & 0x1F) * (1.0f / 31.0f);
            float g = ((p >> 5)  & 0x3F) * (1.0f / 63.0f);
            float b = (p & 0x1F) * (1.0f / 31.0f);
            g_ai_in[0 * AI_INPUT_H * AI_INPUT_W + y * AI_INPUT_W + x] = r;
            g_ai_in[1 * AI_INPUT_H * AI_INPUT_W + y * AI_INPUT_W + x] = g;
            g_ai_in[2 * AI_INPUT_H * AI_INPUT_W + y * AI_INPUT_W + x] = b;
        }
    }
}

/* ──────── 摄像头画面显示到 LCD ──────── */
static void display_camera(void) {
    /* DMA2D 或逐像素复制 CAM→LCD, 缩放 640×480 → 800×480 */
    uint16_t *src = (uint16_t *)CAM_FRAME_BUF_ADDR;
    uint16_t *dst = (uint16_t *)LCD_FRAME_BUF_ADDR;
    for (int y = 0; y < LCD_H; y++) {
        int sy = y * CAM_H / LCD_H;
        for (int x = 0; x < LCD_W; x++) {
            int sx = x * CAM_W / LCD_W;
            dst[y * LCD_W + x] = src[sy * CAM_W + sx];
        }
    }
    /* LTDC 重载层地址 */
    HAL_LTDC_SetAddress(&hltdc, LCD_FRAME_BUF_ADDR, 0);
}

/* ──────── 主接口 ──────── */

void EnergyLabel_Init(void) {
    /* 外设初始化 (CubeMX 已生成) */
    MX_FMC_Init();    /* SDRAM */
    MX_I2C1_Init();
    MX_DCIMI_Init();
    MX_LTDC_Init();
    MX_DMA2D_Init();

    /* X-CUBE-AI 初始化 */
    MX_X_CUBE_AI_Init();

    /* LCD 清屏 */
    lcd_fill_rect(0, 0, LCD_W - 1, LCD_H - 1, CL_BLACK);

    /* 摄像头 */
    camera_init();

    lcd_draw_str(10, 200, "Energy Label Detector", CL_GREEN, CL_BLACK);
    lcd_draw_str(10, 220, "STM32N647 + OV5640", CL_GREEN, CL_BLACK);
}

void EnergyLabel_Process(void) {
    if (!g_frame_ready) return;
    g_frame_ready = 0;
    g_frame_cnt++;

    /* 1. 摄像头画面 → LCD */
    display_camera();

    /* 2. 预处理 → AI 输入 */
    preprocess();

    /* 3. NPU 推理 */
    ai_network_run(g_ai_in, g_ai_out);

    /* 4. 解析 YOLO 输出 */
    parse_output(g_ai_out, g_dets, &g_det_count);

    /* 5. 提取信息 */
    int energy_level = -1, has_defect = 0, defect_names[3] = {0}, defect_n = 0;
    int label_idx = -1, box_idx = -1;
    for (uint16_t i = 0; i < g_det_count; i++) {
        uint16_t c = g_dets[i].class_id;
        if (c <= CLASS_LEVEL_5) { if ((int)c + 1 > energy_level) energy_level = c + 1; }
        else if (c >= CLASS_STAIN && c <= CLASS_WRINKLE && defect_n < 3) {
            defect_names[defect_n++] = c; has_defect = 1;
        }
        else if (c == CLASS_LABEL) label_idx = i;
        else if (c == CLASS_BOX)   box_idx = i;
    }

    /* 6. 位置偏差 */
    int pos_dev = 0; float ox = 0, oy = 0;
    if (label_idx >= 0 && box_idx >= 0) {
        Detection_t *l = &g_dets[label_idx], *b = &g_dets[box_idx];
        if (l->confidence >= CONF_THRESHOLD && b->confidence >= CONF_THRESHOLD) {
            float lcx = (l->x1 + l->x2) / 2.0f, lcy = (l->y1 + l->y2) / 2.0f;
            float bcx = (b->x1 + b->x2) / 2.0f, bcy = (b->y1 + b->y2) / 2.0f;
            float bw = b->x2 - b->x1, bh = b->y2 - b->y1;
            if (bw > 0 && bh > 0) {
                ox = (lcx - bcx) / bw; oy = (lcy - bcy) / bh;
                if (fabs(ox) > TOLERANCE_X || fabs(oy) > TOLERANCE_Y) pos_dev = 1;
                float le = fabs(l->x1 - b->x1) / bw, re = fabs(b->x2 - l->x2) / bw;
                float te = fabs(l->y1 - b->y1) / bh, be = fabs(b->y2 - l->y2) / bh;
                if (le < EDGE_MARGIN || re < EDGE_MARGIN || te < EDGE_MARGIN || be < EDGE_MARGIN) pos_dev = 1;
            }
        }
    }

    /* 7. GPIO */
    if (has_defect || pos_dev) {
        HAL_GPIO_WritePin(LED_GPIO_Port, LED_Pin, GPIO_PIN_SET);
        HAL_GPIO_WritePin(BUZZER_GPIO_Port, BUZZER_Pin, GPIO_PIN_SET);
    } else {
        HAL_GPIO_WritePin(LED_GPIO_Port, LED_Pin, GPIO_PIN_RESET);
        HAL_GPIO_WritePin(BUZZER_GPIO_Port, BUZZER_Pin, GPIO_PIN_RESET);
    }

    /* 8. LCD 绘制检测框和信息 */
    for (uint16_t i = 0; i < g_det_count; i++) {
        uint16_t cl;
        if (g_dets[i].class_id <= CLASS_LEVEL_5) cl = CL_GREEN;
        else if (g_dets[i].class_id >= CLASS_STAIN && g_dets[i].class_id <= CLASS_WRINKLE) cl = CL_RED;
        else if (g_dets[i].class_id == CLASS_LABEL) cl = CL_YELLOW;
        else if (g_dets[i].class_id == CLASS_BOX) cl = CL_BLUE;
        else cl = CL_WHITE;
        lcd_draw_rect(g_dets[i].x1, g_dets[i].y1, g_dets[i].x2, g_dets[i].y2, cl);
    }

    char buf[48];
    /* 左上: 能效等级 */
    snprintf(buf, 48, "Level: %d", energy_level > 0 ? energy_level : -1);
    lcd_draw_str(10, 10, buf, energy_level > 0 ? CL_GREEN : CL_RED, CL_BLACK);

    /* 左上: 位置偏差 */
    snprintf(buf, 48, "Pos: %s (%.3f,%.3f)", pos_dev ? "DEV" : "OK", (double)ox, (double)oy);
    lcd_draw_str(10, 40, buf, pos_dev ? CL_RED : CL_GREEN, CL_BLACK);

    /* 左下: 缺陷 */
    const char *dn[] = {"stain", "damage", "wrinkle"};
    int y0 = LCD_H - 30;
    if (has_defect) {
        for (int i = 0; i < defect_n; i++) {
            snprintf(buf, 48, "DEFECT: %s", dn[defect_names[i] - CLASS_STAIN]);
            lcd_fill_rect(8, y0 - 16, 160, y0, CL_DARKRED);
            lcd_draw_str(10, y0 - 14, buf, CL_WHITE, CL_DARKRED);
            y0 -= 28;
        }
    } else {
        lcd_draw_str(10, y0, "Defects: None", CL_GREEN, CL_BLACK);
    }
}

void EnergyLabel_Deinit(void) {
    HAL_DCMI_Stop(&hdcmi);
    HAL_LTDC_DeInit(&hltdc);
    HAL_GPIO_WritePin(LED_GPIO_Port, LED_Pin, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(BUZZER_GPIO_Port, BUZZER_Pin, GPIO_PIN_RESET);
}
