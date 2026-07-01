#!/bin/bash
# =============================================
# 树莓派部署脚本 — 拷贝模型 + 代码到树莓派
# 在 Windows 上运行 (需要 Git Bash 或 WSL)
# =============================================
# 使用前先修改 PI_HOST 为你的树莓派地址

PI_HOST="pi@10.68.243.28"
PI_DIR="/home/pi/energy_label"

echo "创建远程目录..."
ssh ${PI_HOST} "mkdir -p ${PI_DIR}/models"

echo "拷贝推理代码..."
scp deploy/pi_infer_serial.py ${PI_HOST}:${PI_DIR}/
scp deploy/pi_send.py ${PI_HOST}:${PI_DIR}/

echo "拷贝模型文件..."
# 主模型 (320x320 输入，树莓派上最快)
scp models/best_320.onnx ${PI_HOST}:${PI_DIR}/models/best.onnx

# Box 检测器
scp models/box_detector.onnx ${PI_HOST}:${PI_DIR}/models/

echo ""
echo "=== 部署完成 ==="
echo "在树莓派上运行:"
echo "  cd ${PI_DIR}"
echo "  python3 pi_infer_serial.py"
echo ""
echo "如果要后台运行:"
echo "  screen -S detect"
echo "  python3 pi_infer_serial.py"
echo "  Ctrl+A, D 分离"
