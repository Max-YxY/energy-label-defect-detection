"""后处理器单元测试."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import yaml
from inference import PostProcessor


class TestPostProcessor:
    @classmethod
    def setup_class(cls):
        with open("config.yaml", "r") as f:
            cls.config = yaml.safe_load(f)
        cls.pp = PostProcessor(cls.config)

    def test_extract_energy_level_empty(self):
        assert self.pp._extract_energy_level([]) is None

    def test_extract_energy_level_multiple(self):
        boxes = [
            {"class_id": 0, "class_name": "level_1", "confidence": 0.6, "bbox": [0, 0, 100, 100]},
            {"class_id": 2, "class_name": "level_3", "confidence": 0.95, "bbox": [0, 0, 100, 100]},
        ]
        level = self.pp._extract_energy_level(boxes)
        assert level == 3

    def test_extract_defects_dedup(self):
        boxes = [
            {"class_id": 5, "class_name": "stain", "confidence": 0.7, "bbox": [10, 10, 50, 50]},
            {"class_id": 5, "class_name": "stain", "confidence": 0.9, "bbox": [20, 20, 60, 60]},
            {"class_id": 6, "class_name": "damage", "confidence": 0.85, "bbox": [30, 30, 70, 70]},
        ]
        defects = self.pp._extract_defects(boxes)
        assert len(defects) == 2
        assert defects[0]["confidence"] == 0.9
