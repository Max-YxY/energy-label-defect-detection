# STM32N647 引脚分配总结

## 开发板: ATK-DNN647 (正点原子)

## OV5640 摄像头 (DVP → DCIMI)

| 信号 | GPIO | 备注 |
|------|------|------|
| DCMI_D0 | PD7 | |
| DCMI_D1 | PE6 | 与 USART1_RX 复用 |
| DCMI_D2 | PE0 | |
| DCMI_D3 | PB9 | |
| DCMI_D4 | PE8 | |
| DCMI_D5 | PE5 | 与 USART1_TX 复用 |
| DCMI_D6 | PH9 | |
| DCMI_D7 | PB7 | |
| DCMI_HSYNC | PD0 | |
| DCMI_VSYNC | PB8 | |
| DCMI_PIXCLK | PD5 | |
| DCMI_CAM_XCLK | PB14 | 摄像头主时钟 |
| DCMI_CAM_RST | PQ2 | 摄像头复位 |
| DCMI_CAM_PWDN | PG14 | 摄像头掉电 |

## OV5640 控制 (I2C)

| 信号 | GPIO |
|------|------|
| I2C4_SCL | PE13 |
| I2C4_SDA | PE14 |

## 4.3" RGB LCD (800×480) — LTDC

| 信号 | GPIO |
|------|------|
| LTDC_R0~R7 | PG2, PB1, PB5, PB4, PH4, PA15, PF8, PG9 |
| LTDC_G0~G7 | PG12, PG1, PA1, PA0, PB15, PB12, PB11, PB10 |
| LTDC_B0~B7 | PG15, PA7, PA12, PA11, PA10, PA9, PA8, PA2 |
| LTDC_CLK | PA5 |
| LTDC_HSYNC | PF9 |
| LTDC_VSYNC | PG0 |
| LTDC_DE | PG13 |
| LTDC_BL | PA3 (TIM16_CH1 PWM) |

## TF 卡 (SDMMC1)

| 信号 | GPIO |
|------|------|
| SDMMC1_D0~D3 | PC8, PC9, PC10, PC11 |
| SDMMC1_CK | PC12 |
| SDMMC1_CMD | PH2 |

## 板载外设

| 外设 | GPIO |
|------|------|
| LED0 | PG10 |
| LED1 | PE10 |
| BEEP (蜂鸣器) | PD3 |
| KEY0 | PC6 |
| KEY1 | PD1 |
| KEY2 | PG11 |

## 触摸屏 (I2C)

| 信号 | GPIO |
|------|------|
| I2C2_SCL | PD14 |
| I2C2_SDA | PD4 |
| T_PEN | PB3 |
| T_MISO | PD8 |
| T_CS | PD10 |

## 存储

| 外设 | 接口 |
|------|------|
| HyperRAM | XSPIM_P1 (1.8V) |
| NOR Flash | XSPIM_P2 (1.8V) |
