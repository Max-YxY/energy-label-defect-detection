#!/usr/bin/env python3
"""
树莓派 → STM32N647 串口发送缺陷检测结果

使用: python3 pi_send_defect.py [结果]
     python3 pi_send_defect.py OK            # 无缺陷
     python3 pi_send_defect.py NG:wrinkle    # 有缺陷
     python3 pi_send_defect.py NG:stain
     python3 pi_send_defect.py NG:damage
     python3 pi_send_defect.py NG:box

连续发送模式:
     python3 pi_send_defect.py --loop OK     # 每2秒发一次OK
"""

import serial
import serial.tools.list_ports
import sys
import time
import argparse

# 默认串口配置
DEFAULT_PORT = None  # 自动检测
BAUDRATE = 115200
TIMEOUT = 1


def find_stm32_port():
    """自动查找 STM32 串口 (CH340)"""
    ports = serial.tools.list_ports.comports()
    for p in ports:
        # CH340 VID/PID 通常为 1A86:7523
        if "CH340" in p.description or "1A86" in p.vid:
            return p.device
        # 备用: 任意 USB 串口
        if "USB" in p.description or "UART" in p.description:
            return p.device
    return None


def send_result(port, message):
    """发送结果到 STM32
    
    Args:
        port: 串口对象
        message: 要发送的消息 (会自动添加 \\r\\n)
    """
    data = (message + "\\r\\n").encode('utf-8')
    port.write(data)
    print(f"  [发送] {message}")
    
    # 等待回显（可选）
    time.sleep(0.1)
    if port.in_waiting:
        echo = port.read(port.in_waiting).decode('utf-8', errors='ignore')
        print(f"  [回显] {echo.strip()}")


def main():
    parser = argparse.ArgumentParser(description='树莓派 → STM32N647 串口发送缺陷检测结果')
    parser.add_argument('result', nargs='?', default=None,
                        help='检测结果: OK 或 NG:缺陷类型')
    parser.add_argument('--port', '-p', default=None,
                        help=f'串口设备 (默认自动检测)')
    parser.add_argument('--loop', '-l', action='store_true',
                        help='循环发送模式')
    parser.add_argument('--interval', '-i', type=float, default=2.0,
                        help='循环间隔秒数 (默认2秒)')
    parser.add_argument('--list', '-ls', action='store_true',
                        help='列出所有串口设备')
    
    args = parser.parse_args()
    
    # 列出串口
    if args.list:
        ports = serial.tools.list_ports.comports()
        print("可用串口:")
        for p in ports:
            print(f"  {p.device}: {p.description} (VID={p.vid:04X}, PID={p.pid:04X})")
        return
    
    # 自动检测串口
    port_name = args.port or find_stm32_port()
    if not port_name:
        print("错误: 未找到 STM32 串口!")
        print("请指定 --port 或检查 USB 连接")
        ports = serial.tools.list_ports.comports()
        for p in ports:
            print(f"  发现: {p.device} - {p.description}")
        sys.exit(1)
    
    print(f"连接到 {port_name} @ {BAUDRATE} bps...")
    
    try:
        with serial.Serial(port_name, BAUDRATE, timeout=TIMEOUT) as port:
            print("连接成功!\\n")
            
            if args.loop:
                # 循环发送模式
                if not args.result:
                    print("循环模式需要指定结果")
                    sys.exit(1)
                print(f"循环发送: {args.result} (间隔 {args.interval}s)")
                print("按 Ctrl+C 停止\\n")
                try:
                    while True:
                        send_result(port, args.result)
                        time.sleep(args.interval)
                except KeyboardInterrupt:
                    print("\\n已停止")
            else:
                # 单次发送模式
                if args.result:
                    send_result(port, args.result)
                else:
                    # 交互模式
                    print("输入检测结果 (OK / NG:type), 空行退出:")
                    while True:
                        try:
                            msg = input("> ").strip()
                            if not msg:
                                break
                            send_result(port, msg)
                        except KeyboardInterrupt:
                            break
                            
    except serial.SerialException as e:
        print(f"串口错误: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
