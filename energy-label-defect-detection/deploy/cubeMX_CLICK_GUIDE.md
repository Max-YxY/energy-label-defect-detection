# STM32N647 + OV5640 + 4.3寸屏 — CubeMX 点击操作手册

## Step 1: 新建工程

```
① 双击桌面 STM32CubeMX 图标
② File → New Project
③ 顶部搜索框输入: N647
④ 在结果列表点选: STM32N647
⑤ 右下角点: Next
⑥ 工程名填: EnergyLabel_N647
⑦ 路径选:   E:\energy_label_defect_detection\P\firmware\
⑧ 点 Finish
```

---

## Step 2: 配置时钟

```
① 顶部标签页切换到: Pinout & Configuration
② 左侧栏: System Core → RCC
③ HSE: Crystal/Ceramic Resonator (选这个)
④ 顶部标签切到: Clock Configuration
⑤ 输入 HSE 频率: 24 (看板子晶振)
⑥ 在 HCLK 框输入: 800 → 回车 → 自动布好时钟树
```

---

## Step 3: FMC (SDRAM) — 存帧缓冲

```
① 左侧栏: System Core → FMC
② 右边框图点: SDRAM1 或 SDRAM2 (看板子)
③ 在下方 FMC SDRAM1 Configuration:
   - Bank: 选对应的 bank
   - Column bits: 8
   - Row bits: 12
   - CAS Latency: 2
④ 具体时序参数根据你板子的 SDRAM 型号
```

---

## Step 4: DCIMI (OV5640 摄像头)

```
① 左侧栏: Multimedia → DCMI
② Mode: Enabled
③ 下方 DCMI Configuration:
   - Pixel Clock Polarity: Rising (选这个)
   - HSYNC Polarity: Active Low
   - VSYNC Polarity: Active Low
   - Extended Data Mode: 8-bit
   - Capture Rate: All frames
④ OV5640 引脚分配 (右边框图点引脚):
   - DCMI_HSYNC → 点板子原理图的 HSYNC 引脚
   - DCMI_VSYNC → VSYNC 引脚
   - DCMI_PIXCLK → PCLK 引脚
   - DCMI_D0~D7 → 8条数据线引脚
```

---

## Step 5: I2C (OV5640 控制)

```
① 左侧栏: Connectivity → I2C1 (或板子用的那个)
② Mode: I2C
③ 下方参数:
   - I2C Speed Mode: Standard Mode (100KHz)
   - 其他默认
④ 引脚分配: 右键 SCL/SDA → 选对应引脚
```

---

## Step 6: LTDC (4.3寸 RGB 800×480)

```
① 左侧栏: Multimedia → LTDC
② Mode: Enabled
③ Layer Configuration:
   - 点 Layer1
   - Window Horizontal Start: 0
   - Window Horizontal Stop: 799
   - Window Vertical Start: 0
   - Window Vertical Stop: 479
   - Pixel Format: RGB565
   - Constant Alpha: 255
   - Default Color: Black
④ 下方 Timing:
   - Horizontal Sync: 按屏的 datasheet 填 (一般: 30)
   - Horizontal Back Porch: 16
   - Horizontal Front Porch: 16
   - Vertical Sync: 10
   - Vertical Back Porch: 10
   - Vertical Front Porch: 10
   - Pixel Clock: (选外部时钟或分频)
⑤ LTDC 引脚 (右边图):
   - LTDC_R[0:7] → 按原理图
   - LTDC_G[0:7] → 按原理图
   - LTDC_B[0:7] → 按原理图
   - LTDC_HSYNC
   - LTDC_VSYNC
   - LTDC_DE
   - LTDC_CLK
⑥ 背光引脚: 另一个 GPIO → 输出 High
```

---

## Step 7: GPIO (LED + 蜂鸣器)

```
① 左侧栏: System Core → GPIO
② 右边框图找到你想用的空闲引脚
③ 右键点引脚 → GPIO_Output
④ 右键点另一个 → GPIO_Output
⑤ 改名字(重要! 代码中用到):
   - 在引脚上右键 → Enter User Label
   - 第一个输: LED  → 回车
   - 第二个输: BUZZER → 回车
```

---

## Step 8: X-CUBE-AI 导入模型

```
① 顶部: Software Packs → Select Components
② 弹出的窗口:
   - 搜索: X-CUBE-AI
   - 展开 STMicroelectronics → X-CUBE-AI
   - 勾选: AI → Core → 选版本
   - 点 OK
③ 左侧栏最下方: Software Packs → STMicroelectronics.X-CUBE-AI
④ 中间: Add Network → 弹窗中选:
   - Network type: ONNX
   - File path: 点 Browse → 选 E:\energy_label_defect_detection\P\models\best.onnx
   - 点 OK
⑤ Compression: 点下拉 → 选 INT8 (Neural-ART NPU)
⑥ 点 Analyze (等几秒)
   - 看报告: NPU ratio 应 > 90%
   - 看 RAM: 是否 < 你板子的可用 RAM
⑦ 点 OK 确认
```

---

## Step 9: 生成代码前检查

```
检查这些外设都在 Pinout 页右边绿色了(全部配通):
✅ DCMI       (摄像头数据)
✅ I2C1       (摄像头控制) 
✅ LTDC       (LCD 显示)
✅ FMC        (SDRAM 帧缓冲)
✅ GPIO: LED  (报警灯)
✅ GPIO: BUZZER (蜂鸣器)
✅ X-CUBE-AI  (AI 模型已导入)

❗ 如果有红色的引脚 → 重新分配或删掉冲突的外设
```

---

## Step 10: 生成工程

```
① 顶部菜单: Project → Generate Code
② 弹窗:
   - Toolchain: STM32CubeIDE
   - 点 Generate
③ 生成完后点: Open Project
   → 自动启动 STM32CubeIDE
```

---

## Step 11: 添加应用代码

```
① CubeIDE 工程树:
   - Core → Src → main.c (双击打开)
② 在 main.c 找到这行: /* USER CODE BEGIN Includes */
③ 在它下面一行粘贴:
   #include "app_energy_label.h"
④ 找到: /* USER CODE BEGIN 2 */
   在下面粘贴:
   EnergyLabel_Init();
⑤ 找到: /* USER CODE BEGIN 3 */
   在下面粘贴:
   EnergyLabel_Process();
   HAL_Delay(30);
⑥ 把我们的 C 文件复制进工程:
   - 复制 deploy/stm32n6_app_energy_label.c → Core/Src/
   - 复制 deploy/app_energy_label.h    → Core/Inc/
   - 复制 deploy/ov5640_driver.c       → Core/Src/
   - 复制 deploy/ov5640_driver.h       → Core/Inc/
⑦ 右键工程 → Refresh → 新文件出现
```

---

## Step 12: 添加中断回调

```
① 打开 Core/Src/stm32n6xx_it.c
② 末尾找到: /* USER CODE BEGIN 1 */
③ 在上面加:
void HAL_DCMI_FrameEventCallback(DCMI_HandleTypeDef *hdcmi)
{
    EnergyLabel_FrameReady();
}
④ 在 stm32n6xx_it.c 头部 Includes 位置加:
   #include "app_energy_label.h"
```

---

## Step 13: 编译 & 烧录

```
① 点锤子图标: Build All (或 Ctrl+B)
② 看下方 Console: BUILD SUCCESSFUL
③ ST-Link 接好板子
④ 点绿色三角形: Run (或 F11)
⑤ 第一次烧录会弹 Debug Config → 点 OK
⑥ 烧录完成自动运行
```

---

## 如果编译报错

| 报错 | 原因 | 修法 |
|---|---|---|
| `undefined reference to EnergyLabel_Init` | 没加 .c 文件 | 右键工程 Refresh |
| `ai_network_run` 未定义 | X-CUBE-AI 生成的函数名不对 | 看 `app_x-cube-ai.h` 里实际的函数名 |
| `LCD_FRAME_BUF_ADDR` 地址不对 | SDRAM 映射不同 | 改成你的 SDRAM 地址 |
| `warning: unused variable` | LED/BUZZER 引脚名不同 | 改成你在 CubeMX 命名的引脚名 |
