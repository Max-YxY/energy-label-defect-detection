"""
一键完成：XML→YOLO TXT 转换 + 训练/验证集分割
================================================
来源：all/ 目录（图片和 XML 混放）
目标：data/images/{train,val}/ + data/labels/{train,val}/
用法：
    python scripts/prepare_dataset.py              # 默认 8:2
    python scripts/prepare_dataset.py --ratio 0.15 # 15% 给 val
    python scripts/prepare_dataset.py --seed 123   # 固定随机种子
================================================
"""
import os
import shutil
import random
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

# ===================== 路径配置 =====================
BASE_DIR = Path(__file__).resolve().parent.parent   # P/
SRC_DIR  = BASE_DIR / "all"                         # 原始数据目录
IMG_DIRS = {
    "train": BASE_DIR / "data" / "images" / "train",
    "val":   BASE_DIR / "data" / "images" / "val",
}
LBL_DIRS = {
    "train": BASE_DIR / "data" / "labels" / "train",
    "val":   BASE_DIR / "data" / "labels" / "val",
}

# ===================== 类别定义 =====================
# 顺序即 class_id，已在 XML 中确认的 10 个类别
CLASSES = ["1级", "2级", "3级", "4级", "5级", "标签", "污渍", "破损", "箱子", "褶皱"]

VALID_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


# ===================== XML → YOLO =====================
def parse_xml(xml_path: Path):
    """解析单个 Pascal VOC XML，返回 (img_w, img_h, [(cls_id, x_c, y_c, w, h), ...])"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 读取图片尺寸
    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    annotations = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in CLASSES:
            print(f"  ⚠️  未知类别 '{name}'，已跳过（文件: {xml_path.name}）")
            continue
        cls_id = CLASSES.index(name)
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # YOLO 归一化
        x_c = (xmin + xmax) / 2 / img_w
        y_c = (ymin + ymax) / 2 / img_h
        w   = (xmax - xmin) / img_w
        h   = (ymax - ymin) / img_h

        annotations.append((cls_id, x_c, y_c, w, h))

    return img_w, img_h, annotations


# ===================== 主流程 =====================
def main():
    parser = argparse.ArgumentParser(description="XML→YOLO 转换 + 数据集分割")
    parser.add_argument("--ratio", type=float, default=0.2,
                        help="验证集比例 (默认 0.2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认 42)")
    args = parser.parse_args()

    random.seed(args.seed)

    # ── 1. 扫描源文件 ──────────────────────────────
    all_imgs = []
    for f in SRC_DIR.iterdir():
        if f.suffix.lower() in VALID_IMG_EXTS:
            all_imgs.append(f)

    if not all_imgs:
        print(f"❌ {SRC_DIR} 下未找到图片！")
        return

    total_imgs = len(all_imgs)
    total_xmls = len(list(SRC_DIR.glob("*.xml")))
    print(f"📁 图片总数: {total_imgs}")
    print(f"📁 XML 总数: {total_xmls}")

    # ── 2. 找出有对应 XML 的图片（一一配对）──
    paired   = []   # 有 XML 的图片
    unpaired = []   # 无 XML 的图片（将生成空标注）
    for img in all_imgs:
        xml = SRC_DIR / f"{img.stem}.xml"
        if xml.exists():
            paired.append(img)
        else:
            unpaired.append(img)

    print(f"  ✅ 有标注: {len(paired)} 张")
    print(f"  ⚠️  无标注: {len(unpaired)} 张（将生成空 .txt）")

    # ── 3. 随机分割 ────────────────────────────────
    all_items = list(all_imgs)
    random.shuffle(all_items)

    n_val = max(1, int(len(all_items) * args.ratio))
    val_set = set(all_items[:n_val])
    train_set = set(all_items[n_val:])

    # ── 4. 创建目标目录 ────────────────────────────
    for d in IMG_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)
    for d in LBL_DIRS.values():
        d.mkdir(parents=True, exist_ok=True)

    # ── 5. 逐文件转换 + 复制 ───────────────────────
    stats = {"train": {"img": 0, "lbl": 0, "boxes": 0},
             "val":   {"img": 0, "lbl": 0, "boxes": 0}}

    for img_path in all_items:
        stem = img_path.stem
        split = "val" if img_path in val_set else "train"
        dst_img = IMG_DIRS[split] / img_path.name
        dst_lbl = LBL_DIRS[split] / f"{stem}.txt"

        # 复制图片
        shutil.copy2(img_path, dst_img)
        stats[split]["img"] += 1

        # 生成 YOLO TXT
        xml_path = SRC_DIR / f"{stem}.xml"
        if xml_path.exists():
            _, _, annotations = parse_xml(xml_path)
            lines = [f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
                     for cid, xc, yc, w, h in annotations]
            stats[split]["boxes"] += len(lines)
        else:
            lines = []  # 无标注 → 空文件

        # 写入 TXT（用换行符连接，空列表则创建空文件）
        dst_lbl.write_text("\n".join(lines), encoding="utf-8")
        stats[split]["lbl"] += 1

    # ── 6. 汇报 ─────────────────────────────────────
    print(f"✅ 完成！共转换 {total_imgs} 张图片（种子: {args.seed}）")
    print(f"  类别 ({len(CLASSES)}): {', '.join(CLASSES)}")
    for s in ("train", "val"):
        st = stats[s]
        print(f"  [{s}] 图片: {st['img']} | 标注: {st['lbl']} | 目标框: {st['boxes']}")
    print("=" * 55)
    print("现在运行训练: python main.py train")


if __name__ == "__main__":
    main()
