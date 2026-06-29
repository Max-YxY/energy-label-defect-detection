# Energy Label Defect Detection

Real-time defect detection for energy efficiency labels using YOLOv8n on STM32N647 + Raspberry Pi 4B.  
Detects **5 energy grades** and **3 defect types** (stain / damage / wrinkle) over UART at ~7 FPS.  
Achieves **99.69% accuracy** on 320×320 ONNX models.

---

## Key Features
- Lightweight YOLOv8n deployment on edge devices
- Real-time inference with UART communication
- Multi-class detection: A～E energy grades + 3 surface defects

---

## Performance Metrics
| Metric         | Value          |
|----------------|----------------|
| Model          | YOLOv8n (ONNX) |
| Input Size     | 320×320        |
| Frame Rate     | ~7 FPS         |
| Accuracy       | **99.69%**     |
| Target Platform| STM32N647 + Raspberry Pi 4B |

---

## Detection Results Showcase

Below are sample detection outputs on test images:

| Sample 1 | Sample 2 |
|----------|----------|
| ![Sample1](https://github.com/user-attachments/assets/e6aa289e-72a7-4c38-b122-5416773b6dff) | ![Sample2](https://github.com/user-attachments/assets/62dc67b2-0315-42e8-9067-717e1b2da372) |

| Sample 3 | Sample 4 |
|----------|----------|
| ![Sample3](https://github.com/user-attachments/assets/5842d128-434e-472c-9ca1-8d2e920ca021) | ![Sample4](https://github.com/user-attachments/assets/e4e14b2a-b65a-4223-8f8d-1561eaf1a735) |

| Sample 5 |
|----------|
| ![Sample5](https://github.com/user-attachments/assets/3a610052-c4f1-436d-bea6-8b028be06904) |


---

## Dataset Download

The dataset used in this project is available via Baidu Netdisk:

- **Link**: [https://pan.baidu.com/s/1Ub9j9IpvRajbUlzjbeoSTQ?pwd=1145](https://pan.baidu.com/s/1Ub9j9IpvRajbUlzjbeoSTQ?pwd=1145)  
- **Extraction Code**: `1145`  
- *Shared by Baidu Netdisk Super VIP v3*

---

## Citation / Contact
If you find this work useful, please star ⭐ the repository and feel free to open issues for any questions.
