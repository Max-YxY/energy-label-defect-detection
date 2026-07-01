"""项目安装脚本."""
import sys
from setuptools import setup, find_packages

# 如果直接运行 python setup.py 不带任何参数，打印提示
if __name__ == "__main__" and len(sys.argv) == 1:
    print("=" * 60)
    print("  Energy Label Defect Detection - v2.0.0")
    print("=" * 60)
    print()
    print("  用法:")
    print("    pip install -e .              (推荐) 开发模式安装")
    print("    python setup.py develop        开发模式安装")
    print("    python setup.py install        正式安装")
    print("    python setup.py build          仅构建")
    print()
    print("  安装后可用命令:")
    print("    label-detector                 启动检测训练/推理")
    print("    label-api                      启动 API 服务")
    print("=" * 60)
    sys.exit(0)

setup(
    name="energy_label_detection",
    version="2.0.0",
    description="基于 OpenHarmony 的产品能效标签与缺陷检测系统",
    author="Quality Engineering Team",
    py_modules=['main'],
    packages=find_packages(exclude=["tests", "docs"]),
    install_requires=[
        "torch>=2.0.0",
        "ultralytics>=8.1.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0,<2.0.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "loguru>=0.7.0",
        "flask>=3.0.0",
        "pillow>=10.0.0",
        "albumentations>=1.3.0",
    ],
    entry_points={
        "console_scripts": [
            "label-detector=main:main",
            "label-api=deploy.api_server:main",
        ],
    },
    python_requires=">=3.10",
)
