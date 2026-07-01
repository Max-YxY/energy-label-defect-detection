"""
训练器模块.
"""
import os
import sys
import warnings
from pathlib import Path
import yaml
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import get_logger
from data import create_dataset_yaml

logger = get_logger(__name__)


class LabelDefectTrainer:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.training_cfg = self.config["training"]
        self.model_cfg = self.config["model"]
        self.data_cfg = self.config["data"]

    def run(self, resume: bool = False) -> str:
        dataset_yaml = create_dataset_yaml(
            root_path=self.data_cfg["root"],
            output_path=self.data_cfg.get("dataset_yaml", "./data/dataset.yaml"),
            num_classes=self.model_cfg["num_classes"],
        )
        logger.info(f"Dataset YAML created at: {dataset_yaml}")

        arch = self.model_cfg["architecture"]
        pretrained = self.model_cfg.get("pretrained_path", f"{arch}.pt")
        # If pretrained arch differs from target arch: create target model, transfer weights
        model = None
        if pretrained != arch + ".pt" and os.path.exists(pretrained) and not resume:
            logger.info(f"Creating {arch} from {arch}.yaml, loading {pretrained}")
            model = YOLO(f"{arch}.yaml").load(pretrained)
            logger.info(f"Loaded {pretrained} weights into {arch}")
        elif not resume:
            model = YOLO(pretrained)
            logger.info(f"Starting fresh from {pretrained}")
        else:
            # Resume: load last.pt from latest train dir
            import glob as _glob
            train_dirs = sorted(_glob.glob("./runs/detect/train*"), key=os.path.getmtime, reverse=True)
            latest_dir = train_dirs[0] if train_dirs else None
            last_pt = os.path.join(latest_dir, "weights", "last.pt") if latest_dir else None
            if last_pt and os.path.exists(last_pt):
                model = YOLO(last_pt)
                logger.info(f"Resuming from {last_pt}")
            else:
                logger.error("Cannot resume: last.pt not found")

        import torch as _torch
        _gpu_ok = _torch.cuda.is_available()
        _device_count = _torch.cuda.device_count() if _gpu_ok else 0
        logger.info(f"PyTorch {_torch.__version__} | CUDA available: {_gpu_ok} | devices: {_device_count}")
        training_device = self.training_cfg.get("device", "cpu")
        if _gpu_ok and _device_count > 0:
            _torch.cuda.set_device(0)
            logger.info(f"GPU: {_torch.cuda.get_device_name(0)} | device={training_device}")
        elif _gpu_ok:
            logger.info("CUDA available but 0 devices visible — forcing device=0 for ultralytics")
        else:
            logger.info(f"Running on {training_device}")

        train_args = {
            "data": dataset_yaml,
            "epochs": self.training_cfg["epochs"],
            "patience": self.training_cfg["patience"],
            "batch": self.training_cfg["batch_size"],
            "imgsz": self.model_cfg["input_size"][0],
            "amp": False,
            "optimizer": self.training_cfg["optimizer"],
            "lr0": self.training_cfg["lr0"],
            "momentum": self.training_cfg["momentum"],
            "weight_decay": self.training_cfg["weight_decay"],
            "cls": self.training_cfg.get("cls_weight", 1.5),
            "seed": self.training_cfg["seed"],
            "mosaic": self.training_cfg["mosaic"],
            "mixup": self.training_cfg["mixup"],
            "copy_paste": self.training_cfg["copy_paste"],
            "flipud": self.training_cfg["flipud"],
            "fliplr": self.training_cfg["fliplr"],
            "hsv_h": self.training_cfg["hsv_h"],
            "hsv_s": self.training_cfg["hsv_s"],
            "hsv_v": self.training_cfg["hsv_v"],
            "degrees": self.training_cfg["degrees"],
            "scale": self.training_cfg["scale"],
            "shear": self.training_cfg["shear"],
            "perspective": self.training_cfg["perspective"],
            "cache": self.training_cfg.get("cache", False),
            "workers": self.training_cfg.get("workers", 4),
            "verbose": True,
            "device": training_device,
            "resume": resume,
        }

        # ── Focal Loss 注入: 替换 BCE loss 为 Focal BCE ──
        fl_gamma = self.training_cfg.get("fl_gamma", 0.0)
        class_weights_cfg = self.training_cfg.get("class_weights", {})

        if fl_gamma > 0 or class_weights_cfg:
            num_cls = self.model_cfg["num_classes"]
            cw_list = [class_weights_cfg.get(i, 1.0) for i in range(num_cls)]
            logger.info(
                f"Loss injection: fl_gamma={fl_gamma}, class_weights={cw_list}"
                if fl_gamma > 0 else f"class_weights injected: {cw_list}"
            )

            import torch.nn as nn
            import torch.nn.functional as F

            class _FocalBCE(nn.Module):
                """Focal Loss with optional per-class alpha. Replaces BCEWithLogitsLoss."""
                def __init__(self, gamma, alpha):
                    super().__init__()
                    self.gamma = gamma
                    if alpha is not None:
                        self.register_buffer("alpha", alpha, persistent=False)
                    else:
                        self.alpha = None

                def forward(self, pred, target):
                    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
                    if self.gamma > 0:
                        pt = (-bce).exp()
                        bce = bce * (1 - pt) ** self.gamma
                    if self.alpha is not None:
                        # alpha only weights positive samples; negatives stay at 1.0
                        alpha_t = target * self.alpha.view(1, 1, -1) + (1 - target)
                        bce = bce * alpha_t
                    return bce

            _injected = [False]

            def _inject_focal(trainer):
                """Inject FocalBCE into criterion on first batch when criterion is ready."""
                if _injected[0]:
                    return
                from ultralytics.utils.torch_utils import unwrap_model
                m = unwrap_model(trainer.model)
                if hasattr(m, "criterion") and hasattr(m.criterion, "bce"):
                    _t = __import__("torch")
                    # BCEWithLogitsLoss(reduction='none') has no params — get device from model
                    dev = next(m.parameters()).device
                    a = _t.tensor(cw_list, dtype=_t.float32, device=dev)
                    m.criterion.bce = _FocalBCE(gamma=fl_gamma, alpha=a)
                    _injected[0] = True
                    verb = f"FocalBCE (gamma={fl_gamma})" if fl_gamma > 0 else "WeightedBCE"
                    logger.info(f"{verb} injected into criterion.bce.")

            model.add_callback("on_train_batch_start", _inject_focal)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = model.train(**train_args)

        best_path = str(Path(results.save_dir) / "weights" / "best.pt")
        target_path = self.model_cfg["weights_path"]
        if best_path != target_path:
            import shutil
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy(best_path, target_path)
            logger.info(f"Best weights copied to {target_path}")
        logger.info("Training completed.")
        return target_path
