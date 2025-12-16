import cv2
import sys

def create_csrt_tracker():
    
    print("Đang khởi tạo thuật toán CSRT...")

    if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
        return cv2.legacy.TrackerCSRT_create()
    
    elif hasattr(cv2, 'TrackerCSRT_create'):
        return cv2.TrackerCSRT_create()
    
    else:
        print("LỖI: Không tìm thấy thuật toán CSRT trong thư viện OpenCV.")
        print("Giải pháp: Chạy lệnh 'pip install opencv-contrib-python'")
        sys.exit()

def calculate_iou(boxA, boxB):
    """Tính chỉ số IOU """
    boxA_coords = (boxA[0], boxA[1], boxA[0] + boxA[2], boxA[1] + boxA[3])
    boxB_coords = (boxB[0], boxB[1], boxB[0] + boxB[2], boxB[1] + boxB[3])

    xA = max(boxA_coords[0], boxB_coords[0])
    yA = max(boxA_coords[1], boxB_coords[1])
    xB = min(boxA_coords[2], boxB_coords[2])
    yB = min(boxA_coords[3], boxB_coords[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA_coords[2] - boxA_coords[0] + 1) * (boxA_coords[3] - boxA_coords[1] + 1)
    boxBArea = (boxB_coords[2] - boxB_coords[0] + 1) * (boxB_coords[3] - boxB_coords[1] + 1)

    return interArea / float(boxAArea + boxBArea - interArea)

# --- CẤU HÌNH ---
VIDEO_SOURCE = 'traffic.mp4' 

if __name__ == '__main__':
    # 1. Khởi tạo video
    video = cv2.VideoCapture(VIDEO_SOURCE)
    if not video.isOpened():
        print("Không thể mở video/webcam")
        sys.exit()

    # 2. Khởi tạo Tracker CSRT (Code đã được cố định)
    tracker = create_csrt_tracker()

    # 3. Đọc frame đầu tiên
    ok, frame = video.read()
    if not ok:
        print('Không đọc được file video')
        sys.exit()

    # 4. Chọn vùng tracking
    print("Vui lòng vẽ hình chữ nhật quanh vật thể và nhấn ENTER hoặc SPACE.")
    bbox = cv2.selectROI("Tracking", frame, False)
    
    # Init tracker
    tracker.init(frame, bbox)
    
    fps_list = []

    while True:
        ok, frame = video.read()
        if not ok:
            break
        
        timer = cv2.getTickCount()

        # Update tracker
        ok, bbox_new = tracker.update(frame)

        # Tính FPS
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        fps_list.append(fps)

        if ok:
            # Tracking thành công -> Vẽ khung xanh
            p1 = (int(bbox_new[0]), int(bbox_new[1]))
            p2 = (int(bbox_new[0] + bbox_new[2]), int(bbox_new[1] + bbox_new[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking thất bại -> Báo lỗi đỏ
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Hiển thị thông tin
        cv2.putText(frame, "CSRT Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        cv2.imshow("Tracking", frame)

        # Nhấn ESC để thoát
        if cv2.waitKey(1) & 0xFF == 27:
            break
            
    if len(fps_list) > 0:
        avg_fps = sum(fps_list) / len(fps_list)
        print(f"FPS Trung bình: {avg_fps:.2f}")
    
    video.release()
    cv2.destroyAllWindows()