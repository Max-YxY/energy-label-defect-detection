"""数据库单元测试."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import tempfile
from utils.database import DetectionDatabase


class TestDetectionDatabase:
    def test_insert_and_query(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            db_path = tf.name
        try:
            db = DetectionDatabase(db_path)
            result = {
                "energy_level": 2,
                "defects": [{"defect_type": "stain", "confidence": 0.95, "bbox": [10, 10, 50, 50]}],
                "position_deviation": False,
                "offset_x": 0.01,
                "offset_y": -0.02,
                "inference_time_ms": 15.3,
            }
            ok = db.insert("PROD-001", result)
            assert ok
            records = db.query(product_id="PROD-001")
            assert len(records) == 1
            assert records[0]["energy_level"] == 2
            assert records[0]["position_deviation"] == 0
        finally:
            Path(db_path).unlink(missing_ok=True)
