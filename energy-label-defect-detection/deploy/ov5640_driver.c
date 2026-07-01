/**
  ******************************************************************************
  * @file    ov5640_driver.c
  * @brief   OV5640 摄像头传感器驱动
  *
  * 接口: SCCB (兼容 I2C) + DVP (8-bit 并行)
  * 初始化: 写入寄存器序列配置输出格式/分辨率
  ******************************************************************************
  */

#include "ov5640_driver.h"

/* ─── OV5640 初始化寄存器序列 ───
 * 以下为 RGB565 640×480 的典型配置
 * 完整序列约 200+ 组, 这里列出关键寄存器
 * 实际完整序列可从 OV5640 datasheet 或
 * https://github.com/ArduCAM/ArduCAM 获取
 */

/* 典型初始化寄存器表 (RGB565 VGA) */
static const OV5640_RegEntry_t ov5640_init_seq[] = {
    /* 软件复位 */
    {0x3103, 0x11},
    {0x3008, 0x82},
    /* 等待 50ms (由调用方延时) */
    
    /* 系统时钟配置 */
    {0x3034, 0x1A}, /* PLL */
    {0x3035, 0x11},
    {0x3036, 0x46},
    {0x3037, 0x08},
    
    /* 输出格式: RGB565 */
    {0x4300, 0x61}, /* RGB565 */
    
    /* 分辨率: VGA (640×480) */
    {0x3800, 0x00}, /* HS */
    {0x3801, 0x00},
    {0x3802, 0x00}, /* VS */
    {0x3803, 0x04},
    {0x3804, 0x0A}, /* HW 输出宽高 */
    {0x3805, 0x3F},
    {0x3806, 0x07},
    {0x3807, 0x9F},
    {0x3808, 0x02}, /* 输出宽度: 640 */
    {0x3809, 0x80},
    {0x380A, 0x01}, /* 输出高度: 480 */
    {0x380B, 0xE0},
    {0x380C, 0x07}, /* HTS */
    {0x380D, 0x58},
    {0x380E, 0x01}, /* VTS */
    {0x380F, 0xF4},
    
    /* ISP 配置 */
    {0x3810, 0x00},
    {0x3811, 0x08},
    {0x3812, 0x00},
    {0x3813, 0x04},
    {0x3814, 0x11}, /* 水平镜像 */
    {0x3815, 0x11}, /* 垂直翻转 */
    
    /* 帧率控制 */
    {0x3A00, 0x38},
    {0x3A01, 0x04},
    {0x3A02, 0x00},
    {0x3A03, 0x0C},
    {0x3A04, 0x00},
    {0x3A05, 0x10},
    {0x3A06, 0x00},
    {0x3A07, 0x06},
    {0x3A08, 0x00},
    {0x3A09, 0x10},
    {0x3A0A, 0x00},
    {0x3A0B, 0x08},
    {0x3A0C, 0x00},
    {0x3A0D, 0x04},
    {0x3A0E, 0x00},
    {0x3A0F, 0x08},
    {0x3A10, 0x00},
    {0x3A11, 0x1C},
    {0x3A12, 0x00},
    {0x3A13, 0x06},
    {0x3A14, 0x00},
    {0x3A15, 0x10},
    {0x3A16, 0x00},
    {0x3A17, 0x08},
    {0x3A18, 0x00},
    {0x3A19, 0x06},
    {0x3A1A, 0x00},
    {0x3A1B, 0x10},
    
    /* AEC/AGC */
    {0x3A0F, 0x30},
    {0x3A10, 0x28},
    {0x3A1B, 0x30},
    {0x3A1E, 0x26},
    {0x3A1F, 0x14},
    
    /* 杂项 */
    {0x3503, 0x00},
    {0x3A00, 0x38},
    {0x3A19, 0x7C},
    {0x3A1A, 0x04},
    
    /* 开启输出 */
    {0x3008, 0x02},
};

#define OV5640_INIT_SEQ_SIZE  (sizeof(ov5640_init_seq) / sizeof(ov5640_init_seq[0]))

/* ─── 接口实现 ─── */

static void (*s_i2c_write)(uint8_t dev_addr, uint16_t reg, uint8_t val) = 0;

static void _write_reg(uint16_t reg, uint8_t val) {
    if (s_i2c_write) {
        s_i2c_write(OV5640_ADDR, reg, val);
    }
}

int OV5640_Init(void (*i2c_write)(uint8_t dev_addr, uint16_t reg, uint8_t val)) {
    s_i2c_write = i2c_write;
    
    /* 软件复位 */
    _write_reg(0x3103, 0x11);
    _write_reg(0x3008, 0x82);
    
    /* 调用方需在复位后延时 50ms */
    /* HAL_Delay(50); */
    
    /* 写入初始化序列 */
    for (size_t i = 0; i < OV5640_INIT_SEQ_SIZE; i++) {
        _write_reg(ov5640_init_seq[i].reg, ov5640_init_seq[i].val);
    }
    
    return 0;
}

int OV5640_SetResolution(uint16_t width, uint16_t height) {
    /* 简易分辨率切换 */
    (void)width;
    (void)height;
    /* 实际需根据目标分辨率计算寄存器值 */
    return 0;
}
