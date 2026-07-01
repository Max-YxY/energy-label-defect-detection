"""
Box-only 检测器训练 — 阶段二：在 label crop 内检测 box.
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.chdir(str(Path(__file__).resolve().parent.parent))


def main():
    import yaml
    from ultralytics import YOLO
    from utils.logger import setup_logger, get_logger

    setup_logger()
    logger = get_logger(__name__)

    with open("box_crops_config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    train_cfg = cfg["training"]

    logger.info(f"Box-only training: {model_cfg['architecture']}, {model_cfg['num_classes']} class, imgsz={model_cfg['input_size'][0]}")

    model = YOLO(f"{model_cfg['architecture']}.pt")

    import torch
    logger.info(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

    results = model.train(
        data="data/box_crops/dataset.yaml",
        epochs=train_cfg["epochs"],
        patience=train_cfg["patience"],
        batch=train_cfg["batch_size"],
        imgsz=model_cfg["input_size"][0],
        optimizer=train_cfg["optimizer"],
        lr0=train_cfg["lr0"],
        lrf=train_cfg["lrf"],
        momentum=train_cfg["momentum"],
        weight_decay=train_cfg["weight_decay"],
        cache=train_cfg.get("cache", False),
        workers=train_cfg.get("workers", 4),
        device=train_cfg.get("device", 0),
        seed=train_cfg["seed"],
        verbose=True,
        amp=False,

        mosaic=train_cfg["mosaic"],
        mixup=train_cfg["mixup"],
        degrees=train_cfg["degrees"],
        scale=train_cfg["scale"],
        shear=train_cfg["shear"],
        perspective=train_cfg["perspective"],
        translate=train_cfg["translate"],
        fliplr=train_cfg["fliplr"],
        hsv_h=train_cfg["hsv_h"],
        hsv_s=train_cfg["hsv_s"],
        hsv_v=train_cfg["hsv_v"],
    )

    import shutil
    best_path = str(Path(results.save_dir) / "weights" / "best.pt")
    target = model_cfg["weights_path"]
    os.makedirs(os.path.dirname(target), exist_ok=True)
    shutil.copy(best_path, target)
    logger.info(f"Box detector saved to {target}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
