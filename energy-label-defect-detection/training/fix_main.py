#!/usr/bin/env python3
with open('E:/1111/SoftwarePackage/Projects/99_Defect_Detection/Appli/Core/Src/main.c', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    if 288 <= i <= 293:
        continue
    new_lines.append(line)

# Insert correct line
new_lines.insert(288, '        HAL_UART_Transmit(&huart1, (uint8_t*)"UART OK\\r\\n", 9, 100);\n')

with open('E:/1111/SoftwarePackage/Projects/99_Defect_Detection/Appli/Core/Src/main.c', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print('Fixed')
