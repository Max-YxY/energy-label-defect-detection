"""
模型转换脚本.
"""
import argparse
from pathlib import Path
from typing import Optional
from ultralytics import YOLO
from utils.logger import get_logger

logger = get_logger(__name__)


def export_to_onnx(weights_path: str, output_path: Optional[str] = None) -> str:
    model = YOLO(weights_path)
    if output_path is None:
        output_path = Path(weights_path).with_suffix(".onnx")
    model.export(format="onnx", imgsz=640, simplify=True)
    logger.info(f"ONNX exported to {output_path}")
    return str(output_path)


def export_to_rknn(onnx_path: str, output_path: str, dataset_path: str, quantized: bool = True):
    try:
        from rknn.api import RKNN
    except ImportError:
        logger.error("RKNN-Toolkit2 not installed.")
        raise
    rknn = RKNN(verbose=True)
    rknn.config(
        mean_values=[[0, 0, 0]],
        std_values=[[255, 255, 255]],
        target_platform="rk3588",
        optimization_level=3,
        quantized_dtype="w8a8" if quantized else "fp16",
    )
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        raise RuntimeError("Failed to load ONNX model.")
    ret = rknn.build(do_quantization=quantized, dataset=dataset_path)
    if ret != 0:
        raise RuntimeError("Failed to build RKNN model.")
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        raise RuntimeError("Failed to export RKNN model.")
    rknn.release()
    logger.info(f"RKNN exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Model conversion tool")
    parser.add_argument("--weights", required=True, help="Path to .pt weights")
    parser.add_argument("--format", choices=["onnx", "rknn", "all"], default="all")
    parser.add_argument("--output-dir", default="./models")
    parser.add_argument("--dataset", default="./data/calibration", help="Calibration dataset")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.format in ("onnx", "all"):
        onnx_path = export_to_onnx(args.weights, str(output_dir / "model.onnx"))
    if args.format in ("rknn", "all"):
        onnx_path = onnx_path if 'onnx_path' in dir() else str(output_dir / "model.onnx")
        export_to_rknn(onnx_path, str(output_dir / "model.rknn"), args.dataset)


if __name__ == "__main__":
    main()
