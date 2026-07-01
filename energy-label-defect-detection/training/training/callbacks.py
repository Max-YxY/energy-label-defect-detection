"""训练回调函数."""
import numpy as np
from ultralytics.utils import LOGGER


def on_train_epoch_end(trainer):
    metrics = trainer.metrics
    LOGGER.info(
        f"Epoch {trainer.epoch}: "
        f"mAP50={metrics.get('metrics/mAP50(B)', 0):.4f}, "
        f"mAP50-95={metrics.get('metrics/mAP50-95(B)', 0):.4f}"
    )


def on_fit_epoch_end(trainer):
    results = trainer.validator_class_result
    if results is not None and len(results) > 0:
        pass
