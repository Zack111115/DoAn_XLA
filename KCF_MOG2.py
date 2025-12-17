import cv2
import numpy as np
import math

class TrafficMonitorKCF:
    def __init__(self, video_path):
        self.video_path = video_path
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=True)
        
        self.active_trackers = [] 
        self.track_id_counter = 0

    def create_tracker(self):
        try:
            tracker = cv2.TrackerKCF_create()
        except AttributeError:
            tracker = cv2.legacy.TrackerKCF_create()
        return tracker

    def check_overlap(self, new_box, existing_boxes, threshold=0.3):
        # Kiểm tra xem box mới có trùng với box nào đang được track không 
        x1, y1, w1, h1 = new_box
        box1_area = w1 * h1
        center1 = (x1 + w1//2, y1 + h1//2)

        for exist_box in existing_boxes:
            x2, y2, w2, h2 = exist_box
            
            # Tính khoảng cách tâm để check nhanh
            center2 = (x2 + w2//2, y2 + h2//2)
            dist = math.hypot(center1[0] - center2[0], center1[1] - center2[1])
            
            # Nếu tâm quá gần nhau coi như là cùng 1 vật thể
            if dist < 40: 
                return True
        return False

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Lỗi mở video.")
            return

        print("--- KCF TRAFFIC MONITOR ---")
        print("Nhấn 'q' để thoát.")

        while True:
            ret, frame = cap.read()
            if not ret: break

            timer = cv2.getTickCount()

            # Resize và ROI
            frame = cv2.resize(frame, (960, 540))
            height, width, _ = frame.shape
            roi = frame
            
            # CẬP NHẬT CÁC TRACKER ĐANG CÓ 
            bboxes_currently_tracked = []
            trackers_to_keep = []

            for item in self.active_trackers:
                tracker = item['tracker']
                success, box = tracker.update(roi) 

                if success:
                    x, y, w, h = [int(v) for v in box]
                    item['bbox'] = (x, y, w, h)
                    bboxes_currently_tracked.append((x, y, w, h))
                    trackers_to_keep.append(item)
                    
                    # Vẽ box từ KCF
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2) 
                    cv2.putText(roi, f"KCF ID: {item['id']}", (x, y - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            self.active_trackers = trackers_to_keep

            # PHÁT HIỆN VẬT THỂ MỚI BẰNG MOG2 
            mask = self.bg_subtractor.apply(roi)
            _, mask = cv2.threshold(mask, 250, 255, cv2.THRESH_BINARY)
            mask = cv2.erode(mask, np.ones((5,5), np.uint8), iterations=1) 
            mask = cv2.dilate(mask, np.ones((5,5), np.uint8), iterations=2)

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 2000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    new_box = (x, y, w, h)

                    # KIỂM TRA XUNG ĐỘT
                    # Chỉ tạo tracker mới nếu vật thể này chưa được KCF theo dõi
                    if not self.check_overlap(new_box, bboxes_currently_tracked):
                        
                        # Khởi tạo KCF Tracker mới
                        tracker = self.create_tracker()
                        tracker.init(roi, new_box)
                        
                        self.active_trackers.append({
                            'tracker': tracker,
                            'id': self.track_id_counter,
                            'bbox': new_box
                        })
                        self.track_id_counter += 1
                        
                        # Vẽ box màu xanh lá (dấu hiệu mới phát hiện)
                        cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)


            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            cv2.rectangle(frame, (0, 0), (300, 100), (0, 0, 0), -1) 
            cv2.addWeighted(frame[0:100, 0:300], 0.7, np.zeros((100, 300, 3), dtype="uint8"), 0.3, 0)
            cv2.putText(frame, "Algorithm: MOG2 + KCF", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Active Trackers: {len(self.active_trackers)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("KCF Traffic Monitor", frame)
            # cv2.imshow("Mask", mask) 

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    monitor = TrafficMonitorKCF("traffic.mp4")
    monitor.run()