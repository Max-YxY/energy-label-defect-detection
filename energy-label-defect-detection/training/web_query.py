#!/usr/bin/env python3
"""命令行查询工具."""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.database import DetectionDatabase
from utils.logger import setup_logger, get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Detection log query tool")
    parser.add_argument("--db", default="./logs/detection.db", help="Database path")
    parser.add_argument("--product-id", help="Filter by product ID")
    parser.add_argument("--start", help="Start time (YYYY-MM-DD HH:MM:SS)")
    parser.add_argument("--end", help="End time")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--export", help="Export to CSV file")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    args = parser.parse_args()

    setup_logger()
    db = DetectionDatabase(args.db)

    if args.stats:
        stats = db.get_statistics()
        print("\n=== Detection Statistics ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return

    if args.export:
        path = db.export_csv(
            args.export,
            product_id=args.product_id,
            start_time=args.start,
            end_time=args.end,
        )
        print(f"Exported to {path}")
        return

    records = db.query(
        product_id=args.product_id,
        start_time=args.start,
        end_time=args.end,
        limit=args.limit,
    )

    print(f"\n{len(records)} records found:")
    print("-" * 80)
    for r in records:
        print(f"  ID: {r['id']} | Product: {r['product_id']} | "
              f"Time: {r['timestamp']} | Level: {r['energy_level']} | "
              f"Deviated: {bool(r['position_deviation'])}")
    print("-" * 80)


if __name__ == "__main__":
    main()
