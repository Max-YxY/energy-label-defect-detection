"""
数据库模块.
"""
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from .logger import get_logger

logger = get_logger(__name__)


class DetectionDatabase:
    def __init__(self, db_path: str = "./logs/detection.db", retry_times: int = 3):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.retry_times = retry_times
        self._init_table()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_table(self):
        create_sql = """
        CREATE TABLE IF NOT EXISTS detection_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id TEXT,
            timestamp TEXT DEFAULT (datetime('now', 'localtime')),
            energy_level INTEGER,
            defects TEXT,
            position_deviation INTEGER,
            offset_x REAL,
            offset_y REAL,
            inference_time_ms REAL,
            image_path TEXT
        );
        """
        with self._get_connection() as conn:
            conn.execute(create_sql)
            conn.commit()
        logger.info("Database table initialized.")

    def insert(self, product_id: str, result: Dict, image_path: str = "") -> bool:
        import json
        defects_json = json.dumps(result.get("defects", []), ensure_ascii=False)
        sql = """
        INSERT INTO detection_logs
            (product_id, energy_level, defects, position_deviation,
             offset_x, offset_y, inference_time_ms, image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """
        params = (
            product_id,
            result.get("energy_level"),
            defects_json,
            1 if result.get("position_deviation") else 0,
            result.get("offset_x", 0.0),
            result.get("offset_y", 0.0),
            result.get("inference_time_ms", 0.0),
            image_path,
        )
        for attempt in range(self.retry_times):
            try:
                with self._get_connection() as conn:
                    conn.execute(sql, params)
                    conn.commit()
                return True
            except sqlite3.Error as e:
                logger.warning(f"DB insert attempt {attempt + 1} failed: {e}")
                time.sleep(0.1 * (attempt + 1))
        logger.error("All DB insert retries failed.")
        return False

    def query(
        self,
        product_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        sql = "SELECT * FROM detection_logs WHERE 1=1"
        params = []
        if product_id:
            sql += " AND product_id = ?"
            params.append(product_id)
        if start_time:
            sql += " AND timestamp >= ?"
            params.append(start_time)
        if end_time:
            sql += " AND timestamp <= ?"
            params.append(end_time)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        with self._get_connection() as conn:
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def export_csv(self, output_path: str, **query_kwargs) -> str:
        records = self.query(limit=100000, **query_kwargs)
        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info(f"Exported {len(df)} records to {output_path}")
        return output_path

    def get_statistics(self) -> Dict:
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM detection_logs").fetchone()[0]
            defect_count = conn.execute(
                "SELECT COUNT(*) FROM detection_logs WHERE defects != '[]'"
            ).fetchone()[0]
            deviation_count = conn.execute(
                "SELECT COUNT(*) FROM detection_logs WHERE position_deviation = 1"
            ).fetchone()[0]
        return {
            "total_detections": total,
            "defect_rate": round(defect_count / max(total, 1), 4),
            "deviation_rate": round(deviation_count / max(total, 1), 4),
        }
