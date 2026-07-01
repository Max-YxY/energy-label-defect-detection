"""
能效标签缺陷检测 — PC 串口发送工具
用法:
  python serial_send.py                 交互模式
  python serial_send.py COM7 send OK   一键发送 OK
  python serial_send.py COM7 send NG   一键发送 NG
  python serial_send.py COM7 send CLS  一键清屏
  python serial_send.py COM7 send <文字> 一键发送自定义文字
"""

import sys
import serial
import serial.tools.list_ports
import time

def list_com_ports():
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("! 没有检测到串口")
        return []
    print("\n可用的串口:")
    for i, p in enumerate(ports):
        print(f"  [{i}] {p.device} — {p.description}")
    return ports

def select_port(ports):
    while True:
        try:
            choice = input(f"\n选择串口 [0-{len(ports)-1}]: ").strip()
            idx = int(choice)
            if 0 <= idx < len(ports):
                return ports[idx].device
        except (ValueError, IndexError):
            pass
        print("输入无效，请重新输入")

def send_and_wait(ser, cmd_bytes, label):
    print(f"\n>>> 发送 {label}...")
    ser.write(cmd_bytes)
    timeout = time.time() + 2
    reply = b""
    while time.time() < timeout:
        if ser.in_waiting:
            reply += ser.read(ser.in_waiting)
        else:
            time.sleep(0.05)
    if reply:
        print(f"<<< 收到回应: {reply.decode(errors='replace')}")
    else:
        print("<<< 无回应")
    return reply

def open_port(port_name):
    try:
        ser = serial.Serial(
            port=port_name,
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1
        )
        print(f"✓ 已打开 {port_name} @ 115200 8N1")
        return ser
    except Exception as e:
        print(f"\n! 打开串口失败: {e}")
        return None

def interactive_mode(ser):
    print("\n" + "=" * 50)
    print("  发送命令给 STM32")
    print("  [1] OK    → LCD 绿色 + PASS")
    print("  [2] NG    → LCD 红色 + FAIL")
    print("  [3] 自定义文字 → LCD 蓝色 + 显示文字")
    print("  [4] 清屏  → LCD 黑色")
    print("  [5] 退出")
    print("=" * 50)
    try:
        while True:
            choice = input("\n请输入 [1/2/3/4/5]: ").strip()
            if choice == "1":
                send_and_wait(ser, b"OK\r\n", "OK (PASS)")
            elif choice == "2":
                send_and_wait(ser, b"NG\r\n", "NG (FAIL)")
            elif choice == "3":
                text = input("请输入要显示的文字: ").strip()
                if text:
                    send_and_wait(ser, f"{text}\r\n".encode(), f"[{text}]")
                else:
                    print("! 文字不能为空")
            elif choice == "4":
                send_and_wait(ser, b"CLS\r\n", "清屏")
            elif choice == "5":
                print("退出")
                break
            else:
                print("! 请输入 1-5")
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        ser.close()
        print("串口已关闭")

def main():
    # 命令行模式: serial_send.py COM7 send OK
    if len(sys.argv) >= 4 and sys.argv[2].lower() == "send":
        port_name = sys.argv[1]
        cmd_text = " ".join(sys.argv[3:])
        ser = open_port(port_name)
        if ser:
            cmd_bytes = f"{cmd_text}\r\n".encode()
            send_and_wait(ser, cmd_bytes, cmd_text)
            ser.close()
        return

    # 命令行模式: serial_send.py COM7
    if len(sys.argv) == 2:
        port_name = sys.argv[1]
        ser = open_port(port_name)
        if ser:
            interactive_mode(ser)
        return

    # 交互模式
    print("=" * 50)
    print("  能效标签缺陷检测 — PC 串口发送工具")
    print("=" * 50)
    ports = list_com_ports()
    if not ports:
        input("\n按 Enter 退出...")
        return
    port_name = select_port(ports)
    ser = open_port(port_name)
    if ser:
        interactive_mode(ser)

if __name__ == "__main__":
    main()
