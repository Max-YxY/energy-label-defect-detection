# STM32N647 缺陷检测 — VSCode 开发教程

## 初始化项目

### 1. 修改 CubeMX 输出为 Makefile

打开 CubeMX，载入你的 `.ioc` 工程：

```
① 打开 STM32CubeMX
② File → Open Project → 选 E:\energy_label_defect_detection\P\firmware\EnergyLabel_N647\EnergyLabel_N647.ioc
③ Project Manager → Project:
   - Toolchain / IDE: 下拉选 Makefile
④ 点右上角:
   - Generate Code
```

生成后会得到 `Makefile` 文件，这是编译的入口。

### 2. VSCode 装这些插件

| 插件 | 作用 |
|---|---|
| **C/C++** (ms-vscode.cpptools) | 代码补全、跳转 |
| **Cortex-Debug** | 调试 STM32 |
| **ARM** | ARM 汇编语法 |
| **Makefile Tools** | 自动识别 Makefile 编译 |

### 3. VSCode 配置

在工程根目录建 `.vscode/c_cpp_properties.json`：

```json
{
    "configurations": [{
        "name": "STM32",
        "includePath": [
            "${workspaceFolder}/Core/Inc",
            "${workspaceFolder}/AppliNonSecure/Core/Inc",
            "${workspaceFolder}/Drivers/CMSIS/Device/ST/STM32N6xx/Include",
            "${workspaceFolder}/Drivers/CMSIS/Include",
            "${workspaceFolder}/Drivers/STM32N6xx_HAL_Driver/Inc",
            "${workspaceFolder}/X-CUBE-AI/App",
            "${workspaceFolder}/X-CUBE-AI/Lib"
        ],
        "defines": [
            "USE_HAL_DRIVER",
            "STM32N647xx"
        ],
        "compilerPath": "arm-none-eabi-gcc",
        "cStandard": "c11",
        "intelliSenseMode": "gcc-arm"
    }],
    "version": 4
}
```

### 4. 编译

```bash
# 在工程根目录
make -j4
```

成功生成 `.elf` 和 `.hex`。

### 5. 烧录

CubeProgrammer 烧录：

```bash
"C:\ST\STM32CubeIDE_2.1.1\STM32CubeIDE\plugins\com.st.stm32cube.ide.mcu.externaltools.cubeprogrammer.win32_2.2.400.202601091506\tools\bin\STM32_Programmer_CLI.exe" -c port=SWD -w build/EnergyLabel_N647.hex -v -rst
```

### 6. VSCode 一键编译（可选）

在工程根目录创建 `.vscode/tasks.json`：

```json
{
    "version": "2.0.0",
    "tasks": [{
        "label": "Build",
        "type": "shell",
        "command": "make",
        "args": ["-j4"],
        "group": {"kind": "build", "isDefault": true},
        "problemMatcher": ["$gcc"]
    }, {
        "label": "Flash",
        "type": "shell",
        "command": "STM32_Programmer_CLI",
        "args": [
            "-c", "port=SWD",
            "-w", "build/EnergyLabel_N647.hex",
            "-v", "-rst"
        ]
    }]
}
```

然后按 `Ctrl+Shift+B` 编译，`Ctrl+Shift+P → Run Task → Flash` 烧录。
