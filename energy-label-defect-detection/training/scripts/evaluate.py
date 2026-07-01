"""
模型评估脚本.
"""
import argparse
from ultralytics import YOLO
from utils.logger import get_logger

logger = get_logger(__name__)


def evaluate(weights_path: str, data_yaml: str, imgsz: int = 640, conf: float = 0.25, iou: float = 0.5, workers: int = 0):
    model = YOLO(weights_path)
    metrics = model.val(data=data_yaml, imgsz=imgsz, conf=conf, iou=iou, split="val", verbose=True, workers=workers)
    logger.info("=" * 60)
    logger.info(f"mAP50:      {metrics.box.map50:.4f}")
    logger.info(f"mAP50-95:   {metrics.box.map:.4f}")
    logger.info(f"Precision:  {metrics.box.mp:.4f}")
    logger.info(f"Recall:     {metrics.box.mr:.4f}")
    logger.info("=" * 60)
    if hasattr(metrics.box, "ap_class_index") and metrics.box.ap_class_index is not None:
        logger.info("Per-class AP50:")
        # ap50 is per-class AP at IoU=0.5; fallback to ap (mAP50-95) if unavailable
        if hasattr(metrics.box, "ap50") and metrics.box.ap50 is not None:
            ap_array = metrics.box.ap50.cpu().numpy() if hasattr(metrics.box.ap50, "cpu") else metrics.box.ap50
        else:
            ap_array = metrics.box.ap.cpu().numpy() if hasattr(metrics.box.ap, "cpu") else metrics.box.ap
        ap_indices = metrics.box.ap_class_index.cpu().numpy() if hasattr(metrics.box.ap_class_index, "cpu") else metrics.box.ap_class_index
        cls_names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
        for idx, ap_val in zip(ap_indices, ap_array):
            name = cls_names.get(int(idx), f"class_{idx}")
            logger.info(f"  {name}: {ap_val:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Model evaluation")
    parser.add_argument("--weights", default="./models/best.pt")
    parser.add_argument("--data", default="./data/dataset.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    evaluate(args.weights, args.data, args.imgsz)


if __name__ == "__main__":
    main()
