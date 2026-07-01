"""模型定义与权重管理."""
from pathlib import Path

MODELS_DIR = Path(__file__).parent
DEFAULT_WEIGHTS = MODELS_DIR / "best.pt"

__all__ = ["MODELS_DIR", "DEFAULT_WEIGHTS"]
