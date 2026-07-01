"""
配置加载工具.
"""
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    _validate_config(config)
    return config


def _validate_config(config: Dict[str, Any]):
    required_sections = ["model", "inference", "training", "data"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing config section: [{section}]")
    if "num_classes" not in config["model"]:
        raise ValueError("model.num_classes is required")
    valid_classes = config["model"]["num_classes"]
    if valid_classes not in (9, 10):
        raise ValueError(f"num_classes must be 9 or 10, got {valid_classes}")
