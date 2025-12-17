import cv2
import sys
import time

def calculate_iou(boxA, boxB):
    """Tính IOU để đánh giá độ chính xác"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0

cap = cv2.VideoCapture('traffic.mp4')
tracker = cv2.TrackerKCF_create()

ret, frame = cap.read()
if not ret: sys.exit()
frame = cv2.resize(frame, (960, 540))

print("\n[HƯỚNG DẪN] Vẽ khung quanh vật thể ở xa nhất -> Nhấn ENTER.")
bbox = cv2.selectROI("KCF_Benchmark", frame, False)
tracker.init(frame, bbox)
initial_bbox = bbox

#  CÁC BIẾN LƯU TRỮ KẾT QUẢ 
fps_list = []
iou_list = []
frame_count = 0
fail_count = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.resize(frame, (960, 540))
    frame_count += 1
    
    start_tick = cv2.getTickCount()
    success, bbox_new = tracker.update(frame)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - start_tick)

    if success:
        fps_list.append(fps)
        iou = calculate_iou(bbox_new, initial_bbox)
        iou_list.append(iou)
        
        x, y, w, h = [int(v) for v in bbox_new]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {int(fps)} | IOU: {iou:.2f}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        fail_count += 1
        cv2.putText(frame, "TRACKING FAILURE! Press 's' to re-select", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("KCF_Benchmark", frame)
    
    key = cv2.waitKey(30) & 0xFF
    if key == ord('s'):
        new_bbox = cv2.selectROI("KCF_Benchmark", frame, False)
        if new_bbox != (0,0,0,0):
            tracker = cv2.TrackerKCF_create()
            tracker.init(frame, new_bbox)
            initial_bbox = new_bbox
    elif key == ord('q'):
        break

# HIỂN THỊ KẾT QUẢ RA TERMINAL 
print("\n" + "="*40)
print("       BÁO CÁO BENCHMARK KCF")
print("="*40)
if len(fps_list) > 0:
    avg_fps = sum(fps_list) / len(fps_list)
    avg_iou = sum(iou_list) / len(iou_list)
    success_rate = ((frame_count - fail_count) / frame_count) * 100

    print(f"- Tổng số khung hình xử lý : {frame_count}")
    print(f"- Tốc độ trung bình (FPS)  : {avg_fps:.2f}")
    print(f"- Độ chính xác trung bình (IOU): {avg_iou:.2f}")
    print(f"- Tỷ lệ bám đuổi thành công : {success_rate:.2f}%")
else:
    print("[!] Không có dữ liệu để hiển thị.")
print("="*40 + "\n")

cap.release()
cv2.destroyAllWindows()