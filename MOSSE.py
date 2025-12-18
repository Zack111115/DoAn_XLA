import cv2
import os
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import time  # Để tính FPS

# --- CẤU HÌNH ---
VISUALIZE = True

# --- HÀM ĐỌC ẢNH UTF-8 ---
def read_image_utf8(path):
    try:
        stream = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        return None

# --- KHỞI TẠO MOSSE ---
def create_mosse_tracker():
    try:
        if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerMOSSE_create'):
            return cv2.legacy.TrackerMOSSE_create()
        elif hasattr(cv2, 'TrackerMOSSE_create'):
            return cv2.TrackerMOSSE_create()
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
    root = tk.Tk()
    root.withdraw()
    print(">>> ĐANG CHẠY BENCHMARK MOSSE <<<")
    print("Vui lòng chọn thư mục chứa dataset OTB...")
    
    dataset_path = filedialog.askdirectory(title="Chọn thư mục dataset OTB")
    if not dataset_path: 
        print("Đã hủy chọn.")
        return

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

    first_frame = read_image_utf8(os.path.join(img_folder, image_files[0]))
    if first_frame is None:
        print("Lỗi: Không đọc được ảnh đầu tiên (Kiểm tra đường dẫn tiếng Việt).")
        return

    tracker = create_mosse_tracker()
    if tracker is None:
        messagebox.showerror("Lỗi", "Không thể khởi tạo MOSSE. Cần cài đặt opencv-contrib-python.")
        return

    print("Đang khởi tạo MOSSE...")
    tracker.init(first_frame, tuple(gt_boxes[0]))

    iou_list = []
    fps_list = []  # ← Danh sách lưu FPS mỗi frame

    for i in range(1, n_frames):
        start_time = time.time()  # ← Bắt đầu đo thời gian frame

        frame = read_image_utf8(os.path.join(img_folder, image_files[i]))
        if frame is None: break

        ok, bbox = tracker.update(frame)

        # Tính IOU (đã sửa int để chính xác)
        if ok:
            pred_box = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
            iou = calculate_iou(pred_box, gt_boxes[i])
        else:
            iou = 0.0
        
        iou_list.append(iou)

        elapsed = time.time() - start_time
        if elapsed > 0:
            fps = 1.0 / elapsed
        else:
            fps = 999.0  # Nếu quá nhanh, coi như cực đại (hoặc 0 cũng được)
        fps_list.append(fps)

        # Hiển thị
        if VISUALIZE:
            display_frame = frame.copy()

            # Vẽ box dự đoán MOSSE (xanh dương)
            if ok:
                pred_box = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                p1 = (pred_box[0], pred_box[1])
                p2 = (pred_box[0] + pred_box[2], pred_box[1] + pred_box[3])
                cv2.rectangle(display_frame, p1, p2, (255, 0, 0), 2, 1)

            # Vẽ groundtruth (xanh lá)
            gt = gt_boxes[i]
            cv2.rectangle(display_frame, (gt[0], gt[1]), (gt[0]+gt[2], gt[1]+gt[3]), (0, 255, 0), 2, 1)
            
            # Hiển thị IOU và FPS realtime
            cv2.putText(display_frame, f"MOSSE IOU: {iou:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("MOSSE Benchmark Running...", display_frame)
            
            if cv2.waitKey(1) & 0xFF == 27:
                break
    
    cv2.destroyAllWindows()

    # --- KẾT QUẢ CUỐI ---
    if iou_list:
        avg_iou = sum(iou_list) / len(iou_list)
        avg_fps = sum(fps_list) / len(fps_list)  # ← Tính trung bình FPS

        print(f"\n--- KẾT QUẢ MOSSE ---")
        print(f"Video: {os.path.basename(dataset_path)}")
        print(f"IOU Trung bình: {avg_iou:.4f}")
        print(f"FPS Trung bình: {avg_fps:.1f}")

        # Biểu đồ IOU + thêm thông tin Avg FPS
        plt.figure(figsize=(10, 5))
        plt.plot(iou_list, label=f'MOSSE (Avg IOU: {avg_iou:.3f} | Avg FPS: {avg_fps:.1f})', color='blue', linewidth=2)
        plt.axhline(y=0.5, color='r', linestyle='--', label='Ngưỡng chấp nhận (0.5)')
        
        folder_name = os.path.basename(dataset_path)
        plt.title(f'Biểu đồ độ chính xác MOSSE - Video: {folder_name}')
        plt.xlabel('Frame (Thời gian)')
        plt.ylabel('IOU Score (Độ chính xác)')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    run_benchmark()