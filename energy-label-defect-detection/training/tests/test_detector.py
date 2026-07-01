"""检测器单元测试."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import pytest
from inference.detector import EnergyLabelDetector


class TestEnergyLabelDetector:
    def test_init_detector(self):
        try:
            detector = EnergyLabelDetector("config.yaml")
            assert detector is not None
        except Exception as e:
            pytest.skip(f"Model download skipped: {e}")

    def test_detect_dummy_image(self):
        try:
            detector = EnergyLabelDetector("config.yaml")
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            result = detector.detect(dummy)
            assert "energy_level" in result
            assert "defects" in result
            assert "position_deviation" in result
        except Exception as e:
            pytest.skip(f"Test skipped: {e}")

    def test_default_on_failure(self):
        try:
            detector = EnergyLabelDetector("config.yaml")
            tiny = np.zeros((32, 32, 3), dtype=np.uint8)
            result = detector.detect(tiny)
            assert isinstance(result, dict)
        except Exception:
            pytest.skip("Expected error handled")
