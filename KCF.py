import cv2
import sys
import os
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import time

# --- CẤU HÌNH HỆ THỐNG ---
VISUALIZE = True  
TRACKER_NAME = "KCF (Kernelized Correlation Filters)"

def read_image_utf8(path):
    try:
        stream = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(stream, cv2.IMREAD_COLOR)
    except: return None

def calculate_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[0] + boxA[2], boxB[0] + boxB[2]), min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    unionArea = float(boxA[2] * boxA[3] + boxB[2] * boxB[3] - interArea)
    return interArea / unionArea if unionArea > 0 else 0

def run_kcf_evaluation():
    # 1. Khởi tạo chọn dữ liệu
    root = tk.Tk(); root.withdraw()
    print(f"\n>>> KHỞI CHẠY KIỂM THỬ: {TRACKER_NAME} <<<")
    folder = filedialog.askdirectory(title="Chọn thư mục chứa chuỗi ảnh (Dataset)")
    if not folder: return

    # 2. Tìm file Ground Truth và Ảnh
    gt_path = os.path.join(folder, 'groundtruth_rect.txt')
    img_folder = os.path.join(folder, 'img')
    if not os.path.exists(img_folder): img_folder = folder
    
    try:
        with open(gt_path, 'r') as f:
            # Xử lý các định dạng phân tách khác nhau trong file txt
            gt_boxes = [list(map(int, [float(p) for p in l.replace(',',' ').replace('\t',' ').split()]))[:4] for l in f if l.strip()]
    except:
        messagebox.showerror("Lỗi", "Không tìm thấy hoặc không đọc được file groundtruth_rect.txt"); return

    image_files = sorted([f for f in os.listdir(img_folder) if f.lower().endswith(('.jpg', '.png'))])
    n_frames = min(len(image_files), len(gt_boxes))

    # 3. Cài đặt Tracker
    tracker = cv2.legacy.TrackerKCF_create()
    first_frame = read_image_utf8(os.path.join(img_folder, image_files[0]))
    tracker.init(first_frame, tuple(gt_boxes[0]))

    # Biến lưu trữ kết quả
    results = [] 

    print(f"Đang phân tích {n_frames} khung hình. Vui lòng đợi...")

    for i in range(1, n_frames):
        frame = read_image_utf8(os.path.join(img_folder, image_files[i]))
        if frame is None: break

        start_t = time.perf_counter()
        ok, bbox = tracker.update(frame)
        fps = 1.0 / (time.perf_counter() - start_t)
        
        iou = calculate_iou(bbox, gt_boxes[i]) if ok else 0.0
        results.append({"fps": fps, "iou": iou, "ok": ok})

        if VISUALIZE:
            # Vẽ Bounding Box dự đoán (màu Đỏ) và Ground Truth (Xanh lá)
            color = (0, 255, 0) if ok else (0, 0, 255)
            if ok:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2)
            
            gt = gt_boxes[i]
            cv2.rectangle(frame, (gt[0], gt[1]), (gt[0]+gt[2], gt[1]+gt[3]), (0, 255, 0), 1)
            
            # Label thông tin
            status_txt = "TRACKING" if ok else "LOST"
            cv2.putText(frame, f"Mode: {status_txt} | IOU: {iou:.2f}", (15, 30), 1, 1.2, color, 2)
            cv2.imshow("KCF Benchmark Performance", frame)
            if cv2.waitKey(1) & 0xFF == 27: break

    cv2.destroyAllWindows()

    # --- KẾT QUẢ ---
    if results:
        avg_iou = np.mean([r['iou'] for r in results])
        avg_fps = np.mean([r['fps'] for r in results])
        success_rate = sum(1 for r in results if r['ok']) / len(results) * 100

        print("\n" + "="*45)
        print(f"{'THÔNG SỐ ĐÁNH GIÁ THUẬT TOÁN KCF':^45}")
        print("="*45)
        print(f" Video thử nghiệm   : {os.path.basename(folder)}")
        print(f" Tổng số khung hình : {len(results)}")
        print(f" Tỉ lệ bám đuổi     : {success_rate:.2f}%")
        print(f" Tốc độ xử lý (FPS) : {avg_fps:.2f} frames/sec")
        print(f" Độ chính xác (IOU) : {avg_iou:.4f}")
        print("="*45)


        plt.figure("Phân tích Hiệu năng KCF", figsize=(12, 5))
        plt.subplot(121); plt.plot([r['iou'] for r in results], color='blue')
        plt.title("Biến thiên độ chính xác (IOU)"); plt.grid(True)
        plt.subplot(122); plt.plot([r['fps'] for r in results], color='red')
        plt.title("Biến thiên tốc độ (FPS)"); plt.grid(True)
        plt.show()

if __name__ == "__main__":
    run_kcf_evaluation()