import sys; sys.path.insert(0, r'E:\energy_label_defect_detection\P')
import cv2
from inference.detector import EnergyLabelDetector
det = EnergyLabelDetector(r'E:\energy_label_defect_detection\P\config.yaml')
DEVIATION_DIR = r'C:\Users\ASUS\Desktop\位置偏差good'
for f in ['2026-03-05_18-17-51_583.jpg', 'A02_L1_T04_NOR_027.jpg']:
    img = cv2.imread(DEVIATION_DIR + '\\' + f)
    r = det.detect(img)
    print(f'\n=== {f} ===')
    for b in r['boxes']:
        if b['class_id'] in (8,9):
            print(f"  {b['class_name']}: conf={b['confidence']:.4f}, bbox={b['bbox']}")
    print(f"  dev={r['position_deviation']}  dx={r['offset_x']:.4f}  dy={r['offset_y']:.4f}")
    # Check edge margins
    label_boxes = [b for b in r['boxes'] if b['class_id'] == 8]
    box_boxes = [b for b in r['boxes'] if b['class_id'] == 9]
    if label_boxes and box_boxes:
        lb = label_boxes[0]['bbox']
        bb = box_boxes[0]['bbox']
        bw = bb[2] - bb[0]
        bh = bb[3] - bb[1]
        if bw > 0 and bh > 0:
            left = abs((lb[0] - bb[0]) / bw)
            right = abs((bb[2] - lb[2]) / bw)
            top_ = abs((lb[1] - bb[1]) / bh)
            bottom = abs((bb[3] - lb[3]) / bh)
            print(f"  edge margins: L={left:.6f} R={right:.6f} T={top_:.6f} B={bottom:.6f}")
            print(f"  min edge margin: {min(left, right, top_, bottom):.6f}")
