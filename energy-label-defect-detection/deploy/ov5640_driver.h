#ifndef OV5640_DRIVER_H
#define OV5640_DRIVER_H

#include <stdint.h>

/* OV5640 寄存器配置结构 */
typedef struct {
    uint16_t reg;
    uint8_t  val;
} OV5640_RegEntry_t;

/* 初始化 OV5640 (通过 I2C 写入寄存器序列) */
/* i2c_addr: 通过 I2C_WriteReg 函数句柄传入 */
int OV5640_Init(void (*i2c_write)(uint8_t dev_addr, uint16_t reg, uint8_t val));

/* 设置输出分辨率/格式 */
int OV5640_SetResolution(uint16_t width, uint16_t height);

/* 支持的配置 */
#define OV5640_ADDR        0x3C    /* 7-bit address */

/* 分辨率预设 */
#define OV5640_QVGA        320     /* 320×240 */
#define OV5640_VGA         640     /* 640×480 */
#define OV5640_720P        1280    /* 1280×720 */

/* 输出格式 */
#define OV5640_RGB565      0
#define OV5640_JPEG        1
#define OV5640_YUV422      2

#endif
