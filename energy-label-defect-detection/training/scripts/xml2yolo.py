"""
将 Pascal VOC XML 标注转换为 YOLO TXT 格式
每张图片对应一个同名 .txt，内容：class_id x_center y_center w h（归一化）

用法：
    python scripts/xml2yolo.py                           # 只转 train
    python scripts/xml2yolo.py --all                     # train + val 都转
    python scripts/xml2yolo.py --classes NOR DAM STA WRI  # 自定义类别顺序
"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # P/
LABEL_DIRS = {
    "train": BASE_DIR / "data" / "labels" / "train",
    "val":   BASE_DIR / "data" / "labels" / "val",
}
IMG_DIRS = {
    "train": BASE_DIR / "data" / "images" / "train",
    "val":   BASE_DIR / "data" / "images" / "val",
}

# ====== 类别定义（请按实际标注中的 <name> 标签填写，顺序即 class_id） ======
CLASSES = ["DAM", "NOR", "STA", "WRI"]  # ← 按实际修改！


def convert_xml_to_txt(xml_path, img_w, img_h):
    """解析单个 XML，返回 YOLO 格式的行列表"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []
    for obj in root.findall("object"):
        name = obj.find("name").text.strip()
        if name not in CLASSES:
            print(f"  ⚠ 未知类别 '{name}'，已跳过（文件: {xml_path.name}）")
            continue
        cls_id = CLASSES.index(name)
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        # 归一化 YOLO 格式
        x_c = ((xmin + xmax) / 2) / img_w
        y_c = ((ymin + ymax) / 2) / img_h
        w   = (xmax - xmin) / img_w
        h   = (ymax - ymin) / img_h

        lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
    return lines


def get_image_size(img_dir, stem):
    """尝试从图片获取尺寸（需要 PIL）；失败则用 640x640 默认"""
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        img_path = img_dir / f"{stem}{ext}"
        if img_path.exists():
            try:
                from PIL import Image
                with Image.open(img_path) as im:
                    return im.size  # (w, h)
            except Exception:
                pass
    return (640, 640)  # 默认尺寸


def convert_split(split_name):
    lbl_dir = LABEL_DIRS[split_name]
    img_dir = IMG_DIRS[split_name]

    xml_files = sorted(lbl_dir.glob("*.xml"))
    if not xml_files:
        print(f"  [{split_name}] 无 XML 文件，跳过")
        return 0, 0

    converted, skipped = 0, 0
    for xml_path in xml_files:
        stem = xml_path.stem
        img_w, img_h = get_image_size(img_dir, stem)
        lines = convert_xml_to_txt(xml_path, img_w, img_h)

        # 如果图片存在但无目标，生成空文件
        txt_path = lbl_dir / f"{stem}.txt"
        with open(txt_path, "w") as f:
            f.write("\n".join(lines))
        converted += 1
        if not lines:
            skipped += 1

    return converted, skipped


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="XML → YOLO TXT 批量转换")
    parser.add_argument("--all", action="store_true", help="同时转换 train 和 val")
    args = parser.parse_args()

    splits = ["train", "val"] if args.all else ["train"]

    for split in splits:
        print(f"\n{'='*50}")
        print(f"  正在转换 [{split}] ...")
        n, empty = convert_split(split)
        print(f"  ✅ 转换 {n} 个 XML → TXT（{empty} 个为空标注）")

    print(f"\n🎉 完成！现在可以运行: python main.py train")