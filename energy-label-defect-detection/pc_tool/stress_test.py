"""
快速压力测试 — 只发送，不等待回显
用法: python stress_test.py COM7 --fps 20 --count 100
"""

import sys, serial, time, random, argparse

RESULTS = ["OK", "NG", "STAIN", "Wrinkle", "Damage"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("port", help="COM 口")
    parser.add_argument("--fps", type=float, default=10, help="Hz")
    parser.add_argument("--count", type=int, default=50)
    args = parser.parse_args()

    ser = serial.Serial(args.port, 115200, timeout=0.1)
    interval = 1.0 / args.fps
    sent = 0
    t0 = time.time()

    print(f"压力测试: {args.port} @ {args.fps}Hz, {args.count}条")
    print(f"{'#':>4}  发送   耗时")
    print("-" * 30)

    try:
        for i in range(args.count):
            result = random.choice(RESULTS)
            t1 = time.time()
            ser.write(f"{result}\r\n".encode())
            sent += 1
            elapsed = (time.time() - t1) * 1000
            print(f"{sent:>4}  {result:>8}  {elapsed:>5.0f}ms")

            # 等间隔，不等待回显
            sleep = interval - (time.time() - t1)
            if sleep > 0:
                time.sleep(sleep)

    except KeyboardInterrupt:
        pass

    duration = time.time() - t0
    ser.close()
    print("-" * 30)
    print(f"共发送 {sent} 条, 耗时 {duration:.1f}s, 实际 {sent/duration:.1f} Hz")

if __name__ == "__main__":
    main()
