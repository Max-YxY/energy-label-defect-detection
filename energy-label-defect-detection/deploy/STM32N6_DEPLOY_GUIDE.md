# STM32N647 + OV5640 + 4.3寸RGB屏 部署指南

## 硬件清单

| 硬件 | 接口 |
|---|---|
| STM32N647 开发板 | 主控 |
| 4.3寸 RGB 800×480 LCD | LTDC |
| OV5640 摄像头 | DCIMI + I2C(SCCB) |
| ST-Link | SWD |
| TF 卡 | SDIO/SPI |

## 文件清单

| 文件 | 说明 |
|---|---|
| `models/best.onnx` (11.7 MB) | ✅ YOLOv8n ONNX |
| `deploy/stm32n6_app_energy_label.c` | ✅ 核心应用代码 (HAL 级) |
| `deploy/app_energy_label.h` | ✅ 头文件 |
| `deploy/ov5640_driver.h` | ✅ OV5640 摄像头驱动头 |
| `deploy/ov5640_driver.c` | ✅ OV5640 驱动实现 |

## CubeMX 配置步骤

### Step 1: 新建工程
1. 打开 **STM32CubeMX**
2. **File → New Project** → 搜索 `STM32N647` → 选对应芯片

### Step 2: 配置时钟 (RCC)
- HSE: Crystal/Ceramic Resonator
- 配到最大频率 (N6 系列通常 800MHz)

### Step 3: 配置外设引脚

**DCIMI (OV5640 摄像头)**

| 信号 | 建议引脚 |
|---|---|
| DCIMI_HSYNC | 按板子原理图 |
| DCIMI_VSYNC | 按板子原理图 |
| DCIMI_PIXCLK | 按板子原理图 |
| DCIMI_D[0:7] | 8 位数据线 |

**I2C (OV5640 控制)**

| 信号 | 说明 |
|---|---|
| I2Cx_SCL | SCCB 时钟 |
| I2Cx_SDA | SCCB 数据 |

**LTDC (4.3寸RGB 800×480)**

| 信号 | 说明 |
|---|---|
| LTDC_R[0:7] | 红色数据线 |
| LTDC_G[0:7] | 绿色数据线 |
| LTDC_B[0:7] | 蓝色数据线 |
| LTDC_HSYNC | 行同步 |
| LTDC_VSYNC | 场同步 |
| LTDC_DE | 数据使能 |
| LTDC_CLK | 像素时钟 |

**GPIO**

| 引脚 | 功能 |
|---|---|
| LED_Pin | 报警指示灯 (输出) |
| BUZZER_Pin | 蜂鸣器 (输出) |
| BL_Pin | LCD 背光 (输出) |

**SDIO (TF 卡)**
- 如需保存检测记录，配 SDIO 4-bit 模式
- 可选，暂不影响 YOLO 推理

### Step 4: 配置 FMC (SDRAM)

OV5640 和 LTDC 需要大帧缓冲，配置 FMC 外挂 SDRAM：
- FMC → SDRAM1/2
- 根据板子 SDRAM 型号配时序参数

如果板子无 SDRAM，可用内部 RAM 降分辨率 (320×240) 运行。

### Step 5: X-CUBE-AI 导入模型

1. **Software Packs → Select Components**
2. 勾选 **STMicroelectronics → X-CUBE-AI → AI → Core**
3. 左侧栏 → **Software Packs → STMicroelectronics.X-CUBE-AI**
4. 点 **Add Network** → 选择 `E:\energy_label_defect_detection\P\models\best.onnx`
5. **Compression** 选 **INT8 (Neural-ART NPU)**
6. 点 **Analyze** 看 NPU 利用率
7. **Generate Code**

### Step 6: 添加应用代码

生成工程后：

1. 把以下文件复制到工程：
   - `deploy/stm32n6_app_energy_label.c` → `Core/Src/`
   - `deploy/app_energy_label.h` → `Core/Inc/`
   - `deploy/ov5640_driver.h` → `Core/Inc/`
   - `deploy/ov5640_driver.c` → `Core/Src/`

2. 在 `main.c` 的 `/* USER CODE BEGIN Includes */` 添加：
```c
#include "app_energy_label.h"
```

3. 在 `main.c` 的 `/* USER CODE BEGIN 2 */` 添加：
```c
EnergyLabel_Init();
```

4. 在 `main.c` 的 while(1) 中 `/* USER CODE BEGIN 3 */` 添加：
```c
EnergyLabel_Process();
HAL_Delay(30);
```

### Step 7: 编译 & 烧录

- ST-Link 通过 SWD 连接板子
- 在 CubeIDE 中 **Build → Run**
- 或生成 .hex 后用 STM32_Programmer_CLI 烧录：
```
STM32_Programmer_CLI -c port=SWD -w firmware.hex -v -rst
```
