import cv2
import sys
import os
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np

# --- CẤU HÌNH ---
VISUALIZE = True  # True: Xem video khi chạy | False: Chạy ngầm (nhanh)

# --- HÀM ĐỌC ẢNH UTF-8 (SỬA LỖI ĐƯỜNG DẪN TIẾNG VIỆT) ---
def read_image_utf8(path):
    try:
        # Đọc file thành dữ liệu thô rồi giải mã để tránh lỗi font chữ đường dẫn
        stream = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        return None

# --- HÀM KHỞI TẠO CHỈ DÀNH RIÊNG CHO CSRT ---
def create_csrt_tracker():
    """Chỉ tìm và khởi tạo CSRT, không quan tâm thuật toán khác"""
    try:
        # Ưu tiên tìm trong module legacy (OpenCV mới)
        if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
            return cv2.legacy.TrackerCSRT_create()
        # Tìm trong module chính (OpenCV cũ)
        elif hasattr(cv2, 'TrackerCSRT_create'):
            return cv2.TrackerCSRT_create()
    except:
        return None
    return None

def read_groundtruth(folder_path):
    possible_names = ['groundtruth_rect.txt', 'groundtruth.txt']
    gt_path = None
    for f in os.listdir(folder_path):
        if f in possible_names or (f.endswith('.txt') and 'groundtruth' in f):
            gt_path = os.path.join(folder_path, f)
            break
    
    if not gt_path: return None

    gt_boxes = []
    try:
        with open(gt_path, 'r') as f:
            for line in f:
                line = line.replace('\t', ',').replace(' ', ',')
                parts = line.strip().split(',')
                nums = [float(p) for p in parts if p.strip() != '']
                if len(nums) == 4:
                    gt_boxes.append(list(map(int, nums)))
    except: return None
    return gt_boxes

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea == 0: return 0
    return interArea / unionArea

def find_image_folder(dataset_path):
    candidates = ['img', 'imgs', 'sequence', 'images']
    for c in candidates:
        p = os.path.join(dataset_path, c)
        if os.path.exists(p) and os.path.isdir(p): return p
    return dataset_path

# --- CHƯƠNG TRÌNH CHÍNH ---
def run_benchmark():
    # 1. Chọn thư mục
    root = tk.Tk()
    root.withdraw()
    print(">>> ĐANG CHẠY BENCHMARK CSRT <<<")
    print("Vui lòng chọn thư mục chứa dataset OTB...")
    
    dataset_path = filedialog.askdirectory(title="Chọn thư mục dataset OTB")
    if not dataset_path: 
        print("Đã hủy chọn.")
        return

    # 2. Load Dữ liệu
    gt_boxes = read_groundtruth(dataset_path)
    if not gt_boxes:
        messagebox.showerror("Lỗi", "Không tìm thấy file groundtruth (.txt)")
        return

    img_folder = find_image_folder(dataset_path)
    image_files = sorted([f for f in os.listdir(img_folder) if f.lower().endswith(('.jpg', '.png', '.bmp'))])
    
    if not image_files:
        messagebox.showerror("Lỗi", "Không tìm thấy ảnh!")
        return

    n_frames = min(len(image_files), len(gt_boxes))
    print(f"Số lượng frame cần xử lý: {n_frames}")

    # 3. Khởi tạo CSRT
    first_frame = read_image_utf8(os.path.join(img_folder, image_files[0]))
    if first_frame is None:
        print("Lỗi: Không đọc được ảnh đầu tiên (Kiểm tra đường dẫn tiếng Việt).")
        return

    tracker = create_csrt_tracker()
    if tracker is None:
        messagebox.showerror("Lỗi", "Không thể khởi tạo CSRT. Cần cài đặt opencv-contrib-python.")
        return

    print("Đang khởi tạo CSRT...")
    tracker.init(first_frame, tuple(gt_boxes[0]))

    iou_list = []
    
    # 4. Vòng lặp xử lý
    for i in range(1, n_frames):
        frame = read_image_utf8(os.path.join(img_folder, image_files[i]))
        if frame is None: break

        # Update tracker
        ok, bbox = tracker.update(frame)
        
        # Tính toán IOU
        if ok:
            iou = calculate_iou(bbox, gt_boxes[i])
        else:
            iou = 0.0
        
        iou_list.append(iou)

        # Hiển thị
        if VISUALIZE:
            # Vẽ máy đoán (Xanh dương)
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            
            # Vẽ đáp án (Xanh lá)
            gt = gt_boxes[i]
            cv2.rectangle(frame, (gt[0], gt[1]), (gt[0]+gt[2], gt[1]+gt[3]), (0, 255, 0), 2, 1)
            
            cv2.putText(frame, f"CSRT IOU: {iou:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.imshow("CSRT Benchmark Running...", frame)
            
            if cv2.waitKey(1) & 0xFF == 27: # Nhấn ESC để dừng
                break
    
    cv2.destroyAllWindows()

    # 5. Kết quả & Biểu đồ
    if iou_list:
        avg_iou = sum(iou_list) / len(iou_list)
        print(f"\n--- KẾT QUẢ ---")
        print(f"IOU Trung bình của CSRT: {avg_iou:.4f}")
        
        plt.figure(figsize=(10, 5))
        plt.plot(iou_list, label=f'CSRT (Avg IOU: {avg_iou:.2f})', color='blue', linewidth=2)
        plt.axhline(y=0.5, color='r', linestyle='--', label='Ngưỡng chấp nhận (0.5)')
        
        folder_name = os.path.basename(dataset_path)
        plt.title(f'Biểu đồ độ chính xác CSRT - Video: {folder_name}')
        plt.xlabel('Frame (Thời gian)')
        plt.ylabel('IOU Score (Độ chính xác)')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    run_benchmark()