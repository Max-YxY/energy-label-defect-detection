/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : IMX335 摄像头 → NPU 推理 → LCD 显示
  ******************************************************************************
  */
/* USER CODE END Header */
#include "main.h"

/* USER CODE BEGIN Includes */
#include "./LED/led.h"
#include "./RGBLCD/rgblcd.h"
#include "./IMX335/imx335.h"
#include "postprocess.h"
#include "stai_network.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
/* USER CODE END Includes */

/* Private variables ---------------------------------------------------------*/
DMA2D_HandleTypeDef hdma2d;
I2C_HandleTypeDef hi2c2;
LTDC_HandleTypeDef hltdc;
UART_HandleTypeDef huart1;
XSPI_HandleTypeDef hxspi1;

/* USER CODE BEGIN PV */
static stai_network g_network;              /* NPU 网络句柄 */
static float g_npu_input[640 * 640 * 3];    /* NPU 输入缓冲区 (640x640x3 float32) */
static uint32_t g_frame_count = 0;
static uint32_t g_inference_interval = 5;   /* 每5帧推理一次 */
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
static void MX_GPIO_Init(void);
static void MX_DMA2D_Init(void);
static void MX_LTDC_Init(void);
static void MX_I2C2_Init(void);
static void MX_USART1_UART_Init(void);
static void SystemIsolation_Config(void);
void MX_XSPI1_Init(void);

/* USER CODE BEGIN PFP */
static int init_npu(void);
static void preprocess_frame(void);
static void draw_detections(DetectionResult *result);
static void draw_text(int x, int y, const char *text, uint16_t color);
/* USER CODE END PFP */

/**
  * @brief  初始化 NPU
  */
static int init_npu(void)
{
    stai_return_code ret;
    ret = stai_network_init(&g_network);
    if (ret != STAI_OK) return -1;
    return 0;
}

/**
  * @brief  从 LCD 帧缓冲区预处理图像到 NPU 输入
  *         将 800x480 的 LCD 帧缓冲区缩放到 640x640
  */
static void preprocess_frame(void)
{
    int src_w = rgblcddev.width;   /* 800 */
    int src_h = rgblcddev.height;  /* 480 */
    int dst_size = 640;

    /* 从帧缓冲区中心裁剪正方形区域 */
    int crop_size = (src_w < src_h) ? src_w : src_h;
    int offset_x = (src_w - crop_size) / 2;
    int offset_y = (src_h - crop_size) / 2;

    for (int y = 0; y < dst_size; y++) {
        for (int x = 0; x < dst_size; x++) {
            /* 计算源像素位置（最近邻插值） */
            int src_x = offset_x + (x * crop_size) / dst_size;
            int src_y = offset_y + (y * crop_size) / dst_size;

            /* LCD 帧缓冲区为 RGB565 格式 */
            uint16_t pixel = g_ltdc_lcd_framebuf[src_y * src_w + src_x];

            /* RGB565 → float32 RGB [0,1] */
            int r = (pixel >> 11) & 0x1F;
            int g = (pixel >> 5) & 0x3F;
            int b = pixel & 0x1F;

            /* 5-6-5 扩展到 8-bit 再归一化 */
            g_npu_input[(y * dst_size + x) * 3 + 0] = (float)((r * 527 + 23) >> 6) / 255.0f;  /* R */
            g_npu_input[(y * dst_size + x) * 3 + 1] = (float)((g * 259 + 33) >> 6) / 255.0f;  /* G */
            g_npu_input[(y * dst_size + x) * 3 + 2] = (float)((b * 527 + 23) >> 6) / 255.0f;  /* B */
        }
    }
}

/**
  * @brief  在 LCD 上绘制检测结果
  */
static void draw_detections(DetectionResult *result)
{
    uint16_t w = rgblcddev.width;
    uint16_t h = rgblcddev.height;
    int i;

    /* 绘制所有检测框 */
    for (i = 0; i < result->detection_count; i++) {
        Detection *det = &result->detections[i];
        int cid = det->class_id;
        uint16_t color;

        /* 根据类别选颜色 */
        if (cid >= 0 && cid <= 4)       color = 0x07E0;  /* 绿色-等级 */
        else if (cid >= 5 && cid <= 7)  color = 0xF800;  /* 红色-缺陷 */
        else if (cid == 8)              color = 0xFFE0;  /* 黄色-标签 */
        else if (cid == 9)              color = 0x001F;  /* 蓝色-box */
        else                            color = 0xFFFF;

        /* 约束坐标到屏幕范围 */
        int x1 = det->x1; if (x1 < 0) x1 = 0; if (x1 >= (int)w) x1 = w - 1;
        int y1 = det->y1; if (y1 < 0) y1 = 0; if (y1 >= (int)h) y1 = h - 1;
        int x2 = det->x2; if (x2 < 0) x2 = 0; if (x2 >= (int)w) x2 = w - 1;
        int y2 = det->y2; if (y2 < 0) y2 = 0; if (y2 >= (int)h) y2 = h - 1;

        /* 画矩形框 */
        rgblcd_draw_rectangle(x1, y1, x2, y2, color);

        /* 画标签背景 */
        char label[32];
        snprintf(label, sizeof(label), "%s %.2f",
                 i < 10 ? CLASS_NAMES[i] : "?", det->confidence);
        rgblcd_show_string(x1, y1 > 16 ? y1 - 16 : 0, 120, 16, 16, label, color);
    }

    /* 左上角信息区 */
    char info[64];
    int y = 10;

    /* 能效等级 */
    if (result->energy_level > 0) {
        snprintf(info, sizeof(info), "Level: %d", result->energy_level);
        draw_text(10, y, info, 0x07E0);
        y += 20;
    }

    /* 缺陷 */
    if (result->defect_count > 0) {
        snprintf(info, sizeof(info), "Defects: %s", result->defect_names[0]);
        draw_text(10, y, info, 0xF800);
        y += 20;
        for (i = 1; i < result->defect_count; i++) {
            draw_text(10, y, result->defect_names[i], 0xF800);
            y += 20;
        }
    } else {
        draw_text(10, y, "No Defects", 0x07E0);
        y += 20;
    }

    /* 位置偏差 */
    if (result->pos_deviation) {
        draw_text(10, y, "POS DEVIATED", 0xF800);
    } else {
        draw_text(10, y, "Position OK", 0x07E0);
    }

    /* 刷 D-Cache */
    SCB_CleanInvalidateDCache();
}

/**
  * @brief  在 LCD 上显示文字（带黑色背景）
  */
static void draw_text(int x, int y, const char *text, uint16_t color)
{
    /* 先画黑色背景 */
    int len = strlen(text) * 8;  /* 16号字体每个字符约8像素宽 */
    if (len < 16) len = 16;
    rgblcd_fill(x - 2, y - 2, x + len + 2, y + 16 + 2, BLACK);
    rgblcd_show_string(x, y, len + 4, 18, 16, (char *)text, color);
}

int main(void)
{
    SCB_EnableICache();
    SCB_EnableDCache();
    SystemCoreClockUpdate();
    HAL_Init();

    MX_GPIO_Init();
    MX_DMA2D_Init();
    MX_LTDC_Init();
    MX_I2C2_Init();
    MX_USART1_UART_Init();
    MX_XSPI1_Init();
    SystemIsolation_Config();

    led_init();
    rgblcd_init();
    rgblcd_display_dir(1);

    /* 启动画面 */
    rgblcd_clear(BLUE);
    draw_text(100, 200, "Energy Label Detection", WHITE);
    draw_text(100, 230, "NPU Loading...", YELLOW);

    /* 初始化 NPU */
    if (init_npu() != 0) {
        draw_text(100, 260, "NPU Init Failed!", RED);
        while (1);
    }
    draw_text(100, 260, "NPU Ready!", GREEN);

    /* 初始化摄像头 */
    draw_text(100, 280, "Camera Init...", YELLOW);
    if (imx335_init() != 0) {
        draw_text(100, 300, "Camera Error!", RED);
        while (1);
    }
    draw_text(100, 300, "Camera OK!", GREEN);

    /* 启动摄像头采集（直接写入 LCD 帧缓冲区） */
    imx335_start_capture((uint32_t)g_ltdc_lcd_framebuf);

    rgblcd_clear(BLUE);
    draw_text(100, 200, "Running...", GREEN);

    /* 主循环 */
    while (1)
    {
        /* ISP 后台处理 */
        imx335_isp_background_process();

        g_frame_count++;

        /* 每 N 帧推理一次 */
        if (g_frame_count % g_inference_interval == 0) {
            /* 获取 NPU 输入指针 */
            stai_ptr npu_input_ptr;
            stai_network_get_input(&g_network, &npu_input_ptr, 0);

            /* 预处理：LCD 帧缓冲区 → NPU 输入 */
            preprocess_frame();

            /* 复制到 NPU 输入缓冲区 */
            memcpy((void *)npu_input_ptr, g_npu_input, sizeof(g_npu_input));

            /* 刷 D-Cache（NPU 需要从 RAM 读数据） */
            SCB_CleanInvalidateDCache();

            /* 运行 NPU 推理 */
            stai_network_run(&g_network, STAI_RUN_MODE_DEFAULT);

            /* 刷 D-Cache（CPU 需要读 NPU 输出） */
            SCB_CleanInvalidateDCache();

            /* 获取 NPU 输出 */
            stai_ptr npu_output_ptr;
            stai_network_get_output(&g_network, &npu_output_ptr, 0);

            /* 后处理 */
            DetectionResult result;
            postprocess_yolov8(
                (const float *)npu_output_ptr,
                rgblcddev.width,
                rgblcddev.height,
                0.65f,      /* 置信度阈值 */
                0.45f,      /* IoU 阈值 */
                &result);

            /* 绘制检测结果到 LCD */
            draw_detections(&result);
        }

        LED0_TOGGLE();
        HAL_Delay(10);
    }
}

/* ===== 外设初始化（CubeMX 生成，以下为自动生成代码占位）===== */
static void MX_DMA2D_Init(void)
{
    hdma2d.Instance = DMA2D;
    hdma2d.Init.Mode = DMA2D_M2M_PFC;
    hdma2d.Init.ColorMode = DMA2D_OUTPUT_RGB565;
    hdma2d.Init.OutputOffset = 0;
    hdma2d.LayerCfg[1].InputOffset = 0;
    hdma2d.LayerCfg[1].InputColorMode = DMA2D_INPUT_RGB565;
    hdma2d.LayerCfg[1].AlphaMode = DMA2D_NO_MODIF_ALPHA;
    hdma2d.LayerCfg[1].InputAlpha = 255;
    if (HAL_DMA2D_Init(&hdma2d) != HAL_OK) Error_Handler();
    if (HAL_DMA2D_ConfigLayer(&hdma2d, 1) != HAL_OK) Error_Handler();
}

static void MX_LTDC_Init(void)
{
    LTDC_LayerCfgTypeDef pLayerCfg = {0};
    hltdc.Instance = LTDC;
    hltdc.Init.HSPolarity = LTDC_HSPOLARITY_AL;
    hltdc.Init.VSPolarity = LTDC_VSPOLARITY_AL;
    hltdc.Init.DEPolarity = LTDC_DEPOLARITY_AL;
    hltdc.Init.PCPolarity = LTDC_PCPOLARITY_IPC;
    hltdc.Init.HorizontalSync = 0;
    hltdc.Init.VerticalSync = 0;
    hltdc.Init.AccumulatedHBP = 40;
    hltdc.Init.AccumulatedVBP = 8;
    hltdc.Init.AccumulatedActiveW = 520;
    hltdc.Init.AccumulatedActiveH = 280;
    hltdc.Init.TotalWidth = 525;
    hltdc.Init.TotalHeigh = 288;
    hltdc.Init.Backcolor.Blue = 0;
    hltdc.Init.Backcolor.Green = 0;
    hltdc.Init.Backcolor.Red = 0;
    if (HAL_LTDC_Init(&hltdc) != HAL_OK) Error_Handler();
    pLayerCfg.WindowX0 = 0;
    pLayerCfg.WindowX1 = 480;
    pLayerCfg.WindowY0 = 0;
    pLayerCfg.WindowY1 = 272;
    pLayerCfg.PixelFormat = LTDC_PIXEL_FORMAT_RGB565;
    pLayerCfg.Alpha = 255;
    pLayerCfg.Alpha0 = 0;
    pLayerCfg.BlendingFactor1 = LTDC_BLENDING_FACTOR1_CA;
    pLayerCfg.BlendingFactor2 = LTDC_BLENDING_FACTOR2_CA;
    pLayerCfg.FBStartAdress = 0;
    pLayerCfg.ImageWidth = 480;
    pLayerCfg.ImageHeight = 272;
    pLayerCfg.Backcolor.Blue = 0;
    pLayerCfg.Backcolor.Green = 0;
    pLayerCfg.Backcolor.Red = 0;
    if (HAL_LTDC_ConfigLayer(&hltdc, &pLayerCfg, 0) != HAL_OK) Error_Handler();
}

static void MX_I2C2_Init(void)
{
    hi2c2.Instance = I2C2;
    hi2c2.Init.Timing = 0x10707DBC;
    hi2c2.Init.OwnAddress1 = 0;
    hi2c2.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
    hi2c2.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
    hi2c2.Init.OwnAddress2 = 0;
    hi2c2.Init.OwnAddress2Masks = I2C_OA2_NOMASK;
    hi2c2.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
    hi2c2.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
    if (HAL_I2C_Init(&hi2c2) != HAL_OK) Error_Handler();
    HAL_I2CEx_ConfigAnalogFilter(&hi2c2, I2C_ANALOGFILTER_ENABLE);
}

static void MX_USART1_UART_Init(void)
{
    huart1.Instance = USART1;
    huart1.Init.BaudRate = 115200;
    huart1.Init.WordLength = UART_WORDLENGTH_8B;
    huart1.Init.StopBits = UART_STOPBITS_1;
    huart1.Init.Parity = UART_PARITY_NONE;
    huart1.Init.Mode = UART_MODE_TX_RX;
    huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
    huart1.Init.OverSampling = UART_OVERSAMPLING_16;
    huart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
    huart1.Init.ClockPrescaler = UART_PRESCALER_DIV1;
    huart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
    if (HAL_UART_Init(&huart1) != HAL_OK) Error_Handler();
}

static void MX_GPIO_Init(void) { /* GPIO 初始化由 CubeMX 生成 */ }

static void SystemIsolation_Config(void) { /* RIF 配置由 CubeMX 生成 */ }

void MX_XSPI1_Init(void)
{
    XSPIM_CfgTypeDef sXspiManagerCfg = {0};
    XSPI_HyperbusCfgTypeDef sHyperBusCfg = {0};
    hxspi1.Instance = XSPI1;
    hxspi1.Init.FifoThresholdByte = 4;
    hxspi1.Init.MemoryMode = HAL_XSPI_SINGLE_MEM;
    hxspi1.Init.MemoryType = HAL_XSPI_MEMTYPE_HYPERBUS;
    hxspi1.Init.MemorySize = HAL_XSPI_SIZE_256MB;
    hxspi1.Init.ChipSelectHighTimeCycle = 2;
    hxspi1.Init.FreeRunningClock = HAL_XSPI_FREERUNCLK_DISABLE;
    hxspi1.Init.ClockMode = HAL_XSPI_CLOCK_MODE_0;
    hxspi1.Init.WrapSize = HAL_XSPI_WRAP_32_BYTES;
    hxspi1.Init.ClockPrescaler = 1 - 1;
    hxspi1.Init.SampleShifting = HAL_XSPI_SAMPLE_SHIFT_NONE;
    hxspi1.Init.DelayHoldQuarterCycle = HAL_XSPI_DHQC_DISABLE;
    hxspi1.Init.ChipSelectBoundary = HAL_XSPI_BONDARYOF_NONE;
    hxspi1.Init.MaxTran = 0;
    hxspi1.Init.Refresh = 0;
    hxspi1.Init.MemorySelect = HAL_XSPI_CSSEL_NCS1;
    if (HAL_XSPI_Init(&hxspi1) != HAL_OK) Error_Handler();
    sXspiManagerCfg.nCSOverride = HAL_XSPI_CSSEL_OVR_NCS1;
    sXspiManagerCfg.IOPort = HAL_XSPIM_IOPORT_1;
    sXspiManagerCfg.Req2AckTime = 1;
    HAL_XSPIM_Config(&hxspi1, &sXspiManagerCfg, HAL_XSPI_TIMEOUT_DEFAULT_VALUE);
    sHyperBusCfg.RWRecoveryTimeCycle = 7;
    sHyperBusCfg.AccessTimeCycle = 7;
    sHyperBusCfg.WriteZeroLatency = HAL_XSPI_LATENCY_ON_WRITE;
    sHyperBusCfg.LatencyMode = HAL_XSPI_FIXED_LATENCY;
    HAL_XSPI_HyperbusCfg(&hxspi1, &sHyperBusCfg, HAL_XSPI_TIMEOUT_DEFAULT_VALUE);
}

void Error_Handler(void) { __disable_irq(); while(1); }
