"""
实时缺陷检测模拟 — 连续发送 OK/NG 到 STM32
模拟 PC 检测程序实时发送结果，看 STM32 刷新跟不跟得上

用法:
  python simulate_monitor.py COM7          # 默认 5Hz (每 200ms 一条)
  python simulate_monitor.py COM7 --fps 10  # 10Hz
  python simulate_monitor.py COM7 --fps 20  # 20Hz (最快)
  python simulate_monitor.py COM7 --once    # 只发一轮
"""

import sys
import serial
import serial.tools.list_ports
import time
import random
import argparse

RESULTS = ["OK", "NG", "OK", "OK", "NG", "OK", "STAIN", "Wrinkle", "Damage"]
COLORS = {
    "OK": "绿底 PASS",
    "NG": "红底 FAIL",
    "STAIN": "蓝底 STAIN",
    "Wrinkle": "蓝底 Wrinkle",
    "Damage": "蓝底 Damage",
}

def find_com_port():
    ports = serial.tools.list_ports.comports()
    for p in ports:
        if "CH340" in p.description or "USB-SERIAL" in p.description.upper():
            return p.device
    return None

def main():
    parser = argparse.ArgumentParser(description="实时缺陷检测模拟")
    parser.add_argument("port", nargs="?", help="COM 口 (如 COM7)")
    parser.add_argument("--fps", type=float, default=5, help="发送频率 (Hz), 默认 5")
    parser.add_argument("--once", action="store_true", help="只发一轮")
    parser.add_argument("--count", type=int, default=0, help="发送条数, 0=无限")
    args = parser.parse_args()

    port = args.port or find_com_port()
    if not port:
        print("! 请指定 COM 口，如: python simulate_monitor.py COM7")
        sys.exit(1)

    try:
        ser = serial.Serial(port, 115200, timeout=0.5)
    except Exception as e:
        print(f"! 打开 {port} 失败: {e}")
        sys.exit(1)

    interval = 1.0 / args.fps
    sent = 0
    replied = 0
    start_time = time.time()

    print(f"端口: {port}  @ 115200")
    print(f"频率: {args.fps} Hz (每 {interval*1000:.0f}ms 一条)")
    print(f"模式: {'一轮' if args.once else '持续'}")
    print("-" * 50)
    print(f"{'#':>4}  {'发送':>10}  {'收到回显':>10}  {'耗时':>6}")
    print("-" * 50)

    try:
        while True:
            # 随机选一个检测结果
            result = random.choice(RESULTS)
            cmd = f"{result}\r\n".encode()

            t0 = time.time()
            ser.write(cmd)
            sent += 1

            # 等回显
            reply = b""
            while time.time() - t0 < 1.0:
                if ser.in_waiting:
                    reply += ser.read(ser.in_waiting)
                    time.sleep(0.05)  # 给 STM32 处理时间
                else:
                    if reply:
                        break  # 已有数据就退出
                    time.sleep(0.02)  # 还没数据，再等等

            elapsed = (time.time() - t0) * 1000
            reply_text = reply.decode(errors="replace").strip().replace("\r\n", " | ")
            if reply_text:
                replied += 1

            print(f"{sent:>4}  {result:>10}  {reply_text or '<无回应>':>10}  {elapsed:>5.0f}ms", end="")
            
            # 如果没收到回显，标记一下
            if not reply_text:
                print(" ⚠️", end="")
            print()

            # 一轮模式
            if args.once:
                break

            # 限定条数
            if args.count and sent >= args.count:
                break

            # 等下一个间隔
            sleep_time = interval - (time.time() - t0)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n用户中断")
    
    duration = time.time() - start_time
    ser.close()

    print("-" * 50)
    print(f"共发送 {sent} 条, 收到回显 {replied} 条")
    print(f"耗时 {duration:.1f}s, 实际频率 {sent/duration:.1f} Hz")
    print(f"回应率: {replied/sent*100:.1f}%")

if __name__ == "__main__":
    main()
