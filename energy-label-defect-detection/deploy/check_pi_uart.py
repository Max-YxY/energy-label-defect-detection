#!/usr/bin/env python3
"""
远程检查树莓派 CH340 串口驱动状态
用法: python3 check_pi_uart.py
"""
import subprocess
import sys
import os
import tempfile

PI_HOST = "10.68.243.28"
PI_USER = "pi"
PI_PASS = "yxyyxy"

# 要在树莓派上执行的命令
PI_COMMANDS = """
echo "=== USB设备 ==="
lsusb 2>/dev/null || echo 'lsusb not available'

echo "=== 串口设备 ==="
ls -l /dev/ttyUSB* 2>/dev/null; echo "---"
ls -l /dev/ttyAMA* 2>/dev/null; echo "---"
ls -l /dev/serial/* 2>/dev/null || echo 'No /dev/serial'

echo "=== CH340 内核日志 ==="
dmesg | grep -i -E 'ch34|usb|serial' 2>/dev/null | tail -10 || echo 'No dmesg'

echo "=== Python串口库 ==="
python3 -c "
try:
    import serial
    print('pyserial OK:', serial.__version__)
except:
    print('pyserial NOT installed')
" 2>/dev/null

echo "=== 串口权限 ==="
id
ls -la /dev/ttyUSB* 2>/dev/null
"""


def main():
    # 方法1: 尝试 sshpass
    try:
        cmd = [
            "sshpass", "-p", PI_PASS,
            "ssh", "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            f"{PI_USER}@{PI_HOST}",
            PI_COMMANDS
        ]
        print(f"尝试 sshpass 连接 {PI_HOST}...")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        if r.returncode == 0:
            print(r.stdout)
            return
        else:
            print(f"sshpass 失败: {r.stderr}")
    except FileNotFoundError:
        print("sshpass 未安装")
    except Exception as e:
        print(f"sshpass 错误: {e}")

    # 方法2: 生成一个 expect 脚本
    print("\n尝试 expect 方式...")
    expect_script = f"""#!/usr/bin/expect -f
set timeout 15
spawn ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 {PI_USER}@{PI_HOST} "{PI_COMMANDS}"
expect "password:"
send "{PI_PASS}\\r"
expect eof
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.exp', delete=False) as f:
        f.write(expect_script)
        exp_path = f.name

    try:
        os.chmod(exp_path, 0o755)
        r = subprocess.run(['expect', exp_path], capture_output=True, text=True, timeout=25)
        if r.stdout.strip():
            print(r.stdout)
        if r.stderr.strip():
            print("STDERR:", r.stderr)
    except FileNotFoundError:
        print("expect 未安装")
    except Exception as e:
        print(f"expect 错误: {e}")
    finally:
        os.unlink(exp_path)

    print("\n❌ 无法自动连接树莓派")
    print("请手动在树莓派上运行:")
    print("  lsusb")
    print("  ls -l /dev/ttyUSB*")
    print("  dmesg | grep ch34")


if __name__ == '__main__':
    main()
