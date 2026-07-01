"""
树莓派 → STM32 串口发送检测结果
新协议: L<等级>,<缺陷列表>,<位置>

用法:
  python3 pi_send.py COM7                           交互模式
  python3 pi_send.py COM7 "L3,OK,OK"                单次发送
  python3 pi_send.py COM7 "L2,STAIN+DAMAGE,DEV"     多个缺陷+偏移
  
集成到检测循环:
  from pi_send import send_result
  send_result(ser, energy_level=3, defects=["STAIN"], pos_dev=False)
"""

import serial
import serial.tools.list_ports
import sys
import time


def find_stm32_port():
    ports = serial.tools.list_ports.comports()
    for p in ports:
        if "CH340" in p.description or "1A86" in str(p.vid):
            return p.device
        if "USB" in p.description or "UART" in p.description:
            return p.device
    return None


def build_cmd(energy_level, defects=None, pos_dev=False):
    """
    构建协议字符串
    
    Args:
        energy_level: 1-5 或 None (未知)
        defects: 缺陷列表, 如 ["STAIN"] 或 ["STAIN","DAMAGE"]
        pos_dev: 位置是否偏移
    Returns:
        格式如 "L3,OK,OK" 或 "L2,STAIN+DAMAGE,DEV"
    """
    # 等级
    if energy_level is None:
        level_str = "L?"
    else:
        level_str = f"L{energy_level}"
    
    # 缺陷
    if defects and len(defects) > 0:
        defect_str = "+".join(d.upper() for d in defects)
    else:
        defect_str = "OK"
    
    # 位置
    pos_str = "DEV" if pos_dev else "OK"
    
    return f"{level_str},{defect_str},{pos_str}"


def send_result(ser, energy_level=None, defects=None, pos_dev=False):
    """发送检测结果到 STM32"""
    cmd = build_cmd(energy_level, defects, pos_dev)
    data = (cmd + "\r\n").encode()
    ser.write(data)
    return cmd


def main():
    if len(sys.argv) < 2:
        port = find_stm32_port()
        if not port:
            print("用法: python3 pi_send.py <COM口|命令>")
            print("   python3 pi_send.py COM7              交互模式")
            print("   python3 pi_send.py COM7 \"L3,OK,OK\"   单次")
            sys.exit(1)
    else:
        port = sys.argv[1]
        if port.upper().startswith("COM") or port.startswith("/dev/"):
            pass  # 是串口名
        else:
            port = find_stm32_port()
            if port:
                sys.argv.insert(1, port)
            else:
                print(f"未找到串口, 请指定: python3 pi_send.py COM7 ...")
                sys.exit(1)

    try:
        ser = serial.Serial(port, 115200, timeout=1)
        print(f"✓ 已连接 {port}")
    except Exception as e:
        print(f"! 打开 {port} 失败: {e}")
        sys.exit(1)

    if len(sys.argv) >= 3:
        # 命令行模式
        cmd = sys.argv[2]
        ser.write(f"{cmd}\r\n".encode())
        time.sleep(0.2)
        reply = b""
        while ser.in_waiting:
            reply += ser.read(ser.in_waiting)
        print(f"  发: {cmd}")
        print(f"  收: {reply.decode(errors='replace').strip()}")
        ser.close()
        return

    # 交互模式
    print("\n输入命令, 空行退出")
    print("示例:")
    print("  L3,OK,OK              Level 3, 无缺陷, 正常")
    print("  L2,STAIN,DEV          Level 2, 污渍, 偏移")
    print("  L1,STAIN+DAMAGE,OK    Level 1, 污渍+破损, 正常")
    print("  OK                    旧协议兼容")
    print("  NG                    旧协议兼容")
    print()

    try:
        while True:
            msg = input("> ").strip()
            if not msg:
                break
            ser.write(f"{msg}\r\n".encode())
            time.sleep(0.2)
            reply = b""
            while ser.in_waiting:
                reply += ser.read(ser.in_waiting)
            r = reply.decode(errors='replace').strip()
            print(f"  {r}")
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        print("已关闭")


if __name__ == "__main__":
    main()
