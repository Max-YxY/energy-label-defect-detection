"""
最优置信度与IoU阈值搜索脚本.

功能：
1. 在验证集上对 conf / iou 做网格搜索
2. 记录整体 mAP50、Precision、Recall 和逐类 AP50
3. 输出 Top-N 组合，自动更新 config.yaml 中的 inference 阈值
4. 支持按不同指标排序（mAP50 / 小类召回率等）

用法:
    python scripts/find_optimal_thresholds.py                  # 默认网格搜索
    python scripts/find_optimal_thresholds.py --sort damage    # 按 damage 的 AP50 排序
    python scripts/find_optimal_thresholds.py --conf 0.2,0.3,0.4 --iou 0.4,0.5,0.6  # 自定义搜索范围
"""

import argparse
import itertools
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import yaml

# 将项目根目录加入 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
from utils.logger import get_logger

logger = get_logger(__name__)

# ── 默认搜索空间 ─────────────────────────────────────────────
DEFAULT_CONF_RANGE = [round(x, 2) for x in np.arange(0.10, 0.65, 0.05)]  # 0.10 ~ 0.60
DEFAULT_IOU_RANGE  = [round(x, 2) for x in np.arange(0.35, 0.85, 0.05)]  # 0.35 ~ 0.80

# 需要重点关注的弱势类别（项目中的缺陷类 + box）
WEAK_CLASSES = {6: "damage", 7: "wrinkle", 9: "box"}

# ── 核心函数 ──────────────────────────────────────────────────

def evaluate_once(
    model: YOLO,
    data_yaml: str,
    imgsz: int,
    conf: float,
    iou: float,
) -> dict:
    """单次评估，返回关键指标的字典。"""
    metrics = model.val(data=data_yaml, imgsz=imgsz, conf=conf, iou=iou,
                        split="val", verbose=False, plots=False)

    result = {
        "conf": conf,
        "iou": iou,
        "mAP50": float(metrics.box.map50),
        "mAP50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "per_class_ap50": {},
    }

    # ── 安全转换为 numpy 数组（兼容 tensor / numpy） ──
    def to_numpy(x):
        """安全转换为 numpy 数组（兼容 tensor / numpy）。"""
        if x is None:
            return np.array([])
        if hasattr(x, 'cpu'):
            return x.cpu().numpy()
        if isinstance(x, np.ndarray):
            return x
        return np.array(x)

    # 优先使用 ap50，不存在则回退到 ap
    if hasattr(metrics.box, "ap50") and metrics.box.ap50 is not None:
        ap_array = to_numpy(metrics.box.ap50)
    elif hasattr(metrics.box, "ap") and metrics.box.ap is not None:
        ap_array = to_numpy(metrics.box.ap)  # fallback
    else:
        ap_array = np.array([])

    ap_indices = to_numpy(
        getattr(metrics.box, "ap_class_index", None)
    )

    cls_names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    for idx, ap_val in zip(ap_indices, ap_array):
        name = cls_names.get(int(idx), f"class_{idx}")
        result["per_class_ap50"][name] = float(ap_val)

    return result


def grid_search(
    weights_path: str,
    data_yaml: str,
    imgsz: int,
    conf_range: List[float],
    iou_range: List[float],
) -> List[dict]:
    """遍历所有 conf × iou 组合，返回结果列表。"""
    model = YOLO(weights_path)
    results: List[dict] = []
    total = len(conf_range) * len(iou_range)

    logger.info(f"开始网格搜索: {len(conf_range)} conf × {len(iou_range)} iou = {total} 组合")

    for idx, (conf, iou) in enumerate(itertools.product(conf_range, iou_range), 1):
        try:
            r = evaluate_once(model, data_yaml, imgsz, conf, iou)
            results.append(r)
            # 简短一行摘要
            weak_aps = ", ".join(
                f"{WEAK_CLASSES.get(cid, cid)}={r['per_class_ap50'].get(WEAK_CLASSES[cid], 0):.3f}"
                for cid in WEAK_CLASSES
            )
            logger.info(
                f"[{idx:3d}/{total}] conf={conf:.2f} iou={iou:.2f} → "
                f"mAP50={r['mAP50']:.4f} | P={r['precision']:.4f} R={r['recall']:.4f} | {weak_aps}"
            )
        except Exception as e:
            logger.warning(f"[{idx:3d}/{total}] conf={conf:.2f} iou={iou:.2f} FAILED: {e}")

    return results


def print_top_n(results: List[dict], n: int = 10, sort_by: str = "mAP50"):
    """打印 Top-N 结果。"""
    # 确定排序键
    if sort_by in WEAK_CLASSES.values():
        key_fn = lambda r: r["per_class_ap50"].get(sort_by, 0)
    elif sort_by == "min_weak":
        key_fn = lambda r: min(
            r["per_class_ap50"].get(WEAK_CLASSES[cid], 0) for cid in WEAK_CLASSES
        )
    else:
        key_fn = lambda r: r.get(sort_by, 0)

    sorted_results = sorted(results, key=key_fn, reverse=True)

    print("\n" + "=" * 100)
    print(f"Top-{n} 组合（按 {sort_by} 降序）")
    print("=" * 100)
    header = (
        f"{'Rank':<5} {'conf':<7} {'iou':<7} {'mAP50':<9} {'mAP50-95':<10} "
        f"{'Precision':<10} {'Recall':<10} "
        + " ".join(f"{WEAK_CLASSES[cid]:<9}" for cid in WEAK_CLASSES)
    )
    print(header)
    print("-" * 100)

    for rank, r in enumerate(sorted_results[:n], 1):
        weak_str = " ".join(
            f"{r['per_class_ap50'].get(WEAK_CLASSES[cid], 0):<9.4f}" for cid in WEAK_CLASSES
        )
        line = (
            f"{rank:<5} {r['conf']:<7.2f} {r['iou']:<7.2f} "
            f"{r['mAP50']:<9.4f} {r['mAP50_95']:<10.4f} "
            f"{r['precision']:<10.4f} {r['recall']:<10.4f} "
            + weak_str
        )
        print(line)

    print("-" * 100)
    best = sorted_results[0]
    print(f"\n🏆 最佳组合: conf={best['conf']:.2f}, iou={best['iou']:.2f}")
    print(f"   mAP50={best['mAP50']:.4f}  Precision={best['precision']:.4f}  Recall={best['recall']:.4f}")
    return best


def update_config_yaml(config_path: str, best_conf: float, best_iou: float):
    """将最佳阈值写回 config.yaml。"""
    path = Path(config_path)
    if not path.exists():
        logger.warning(f"config.yaml 不存在: {config_path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("inference", {})["conf_threshold"] = round(best_conf, 2)
    cfg.setdefault("inference", {})["iou_threshold"] = round(best_iou, 2)

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    logger.info(f"已更新 config.yaml: conf_threshold={best_conf:.2f}, iou_threshold={best_iou:.2f}")


def save_full_results(results: List[dict], output_path: str):
    """保存完整网格搜索结果到 CSV。"""
    csv_lines = ["rank,conf,iou,mAP50,mAP50-95,precision,recall," +
                 ",".join(WEAK_CLASSES[cid] for cid in WEAK_CLASSES)]
    sorted_by_map = sorted(results, key=lambda r: r["mAP50"], reverse=True)
    for rank, r in enumerate(sorted_by_map, 1):
        weak_vals = ",".join(
            f"{r['per_class_ap50'].get(WEAK_CLASSES[cid], 0):.4f}" for cid in WEAK_CLASSES
        )
        csv_lines.append(
            f"{rank},{r['conf']:.2f},{r['iou']:.2f},"
            f"{r['mAP50']:.4f},{r['mAP50_95']:.4f},"
            f"{r['precision']:.4f},{r['recall']:.4f},{weak_vals}"
        )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(csv_lines))
    logger.info(f"完整结果已保存至 {output_path}")


# ── 主入口 ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="网格搜索最优置信度与 IoU 阈值",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/find_optimal_thresholds.py
  python scripts/find_optimal_thresholds.py --sort damage
  python scripts/find_optimal_thresholds.py --sort min_weak
  python scripts/find_optimal_thresholds.py --conf 0.2,0.3,0.4 --iou 0.4,0.5,0.6
        """,
    )
    parser.add_argument("--weights", default="./models/best.pt", help="模型权重路径")
    parser.add_argument("--data", default="./data/dataset.yaml", help="dataset.yaml 路径")
    parser.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    parser.add_argument(
        "--conf",
        default=",".join(f"{x:.2f}" for x in DEFAULT_CONF_RANGE),
        help="置信度搜索范围，逗号分隔，如 0.1,0.2,0.3",
    )
    parser.add_argument(
        "--iou",
        default=",".join(f"{x:.2f}" for x in DEFAULT_IOU_RANGE),
        help="IoU 搜索范围，逗号分隔，如 0.3,0.4,0.5",
    )
    parser.add_argument(
        "--sort",
        default="mAP50",
        help="排序指标: mAP50（默认）, precision, recall, damage, wrinkle, box, min_weak",
    )
    parser.add_argument("--top", type=int, default=10, help="输出 Top-N 组合")
    parser.add_argument("--update-config", action="store_true",
                        help="自动更新 config.yaml 中 inference 阈值")
    parser.add_argument("--output-csv", default="./grid_search_results.csv",
                        help="完整网格搜索结果 CSV 路径")
    args = parser.parse_args()

    # 解析搜索范围
    conf_range = [float(x.strip()) for x in args.conf.split(",") if x.strip()]
    iou_range = [float(x.strip()) for x in args.iou.split(",") if x.strip()]

    logger.info(f"权重: {args.weights}")
    logger.info(f"数据: {args.data}")
    logger.info(f"Conf 范围: {conf_range}")
    logger.info(f"IoU 范围: {iou_range}")
    logger.info(f"排序指标: {args.sort}")

    # 执行网格搜索
    results = grid_search(args.weights, args.data, args.imgsz, conf_range, iou_range)

    if not results:
        logger.error("无有效结果，请检查模型和数据路径。")
        sys.exit(1)

    # 输出 Top-N
    best = print_top_n(results, n=args.top, sort_by=args.sort)

    # 保存完整结果
    save_full_results(results, args.output_csv)

    # 更新 config.yaml
    if args.update_config:
        update_config_yaml("config.yaml", best["conf"], best["iou"])
    else:
        logger.info("提示: 加 --update-config 可自动将最佳阈值写入 config.yaml")


if __name__ == "__main__":
    main()