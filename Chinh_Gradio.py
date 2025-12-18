import cv2
import gradio as gr
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
import os

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea + 1e-9)

def on_video_change():
    return None, [], pd.DataFrame(columns=["Thuật toán", "FPS TB", "IOU TB", "Tỉ lệ Thành công"]), None

def get_first_frame(video_path):
    if not video_path: return None
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret: return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

def run_tracking(video_path, roi_image, algo_name, history):
    if not video_path or roi_image is None:
        return None, pd.DataFrame(history), None, "Vui lòng chọn video và khoanh vùng vật thể!"

    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(3)), int(cap.get(4))
    fps_orig = cap.get(5)
    
    target = cv2.legacy if hasattr(cv2, 'legacy') else cv2
    tracker_map = {"KCF": target.TrackerKCF_create, "CSRT": target.TrackerCSRT_create, "MOSSE": target.TrackerMOSSE_create}
    tracker = tracker_map[algo_name]()

    template = cv2.cvtColor(roi_image, cv2.COLOR_RGB2GRAY)
    tw, th = template.shape[1], template.shape[0]
    
    # Đọc frame đầu tiên để init tracker
    ret, frame = cap.read()
    if not ret: return None, pd.DataFrame(history), None, "Lỗi đọc video!"
    
    res = cv2.matchTemplate(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(res)
    tracker.init(frame, (max_loc[0], max_loc[1], tw, th))

    # Đặt tên file video duy nhất để tránh lỗi cache của Gradio
    out_path = f"output_{algo_name}_{int(time.time())}.mp4"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'avc1'), fps_orig, (w, h))

    fps_list, iou_list, success_count = [], [], 0
    processed_count = 0

    # THAY ĐỔI TẠI ĐÂY: Chạy hết video gốc thay vì dừng ở 100 frame
    while True:
        ret, frame = cap.read()
        if not ret: break # Thoát khi hết video
        
        # Ground Truth giả lập bằng Template Matching
        res_gt = cv2.matchTemplate(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), template, cv2.TM_CCOEFF_NORMED)
        _, _, _, m_loc = cv2.minMaxLoc(res_gt)
        gt_box = (m_loc[0], m_loc[1], tw, th)

        t_start = time.perf_counter()
        ok, bbox = tracker.update(frame)
        t_end = time.perf_counter()

        fps = 1.0 / (t_end - t_start + 1e-9)
        iou = calculate_iou(bbox, gt_box) if ok else 0
        
        fps_list.append(fps); iou_list.append(iou)
        if ok:
            success_count += 1
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (0, 255, 0), 3)
        
        out.write(frame)
        processed_count += 1

    cap.release(); out.release()

    # Thống kê kết quả
    avg_fps, avg_iou = np.mean(fps_list), np.mean(iou_list)
    history.append([algo_name, f"{avg_fps:.1f}", f"{avg_iou:.3f}", f"{(success_count/processed_count)*100:.1f}%"])
    df_result = pd.DataFrame(history, columns=["Thuật toán", "FPS TB", "IOU TB", "Tỉ lệ Thành công"])

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 4))
    plt.subplot(121); plt.plot(iou_list, color='green'); plt.title(f"IOU: {algo_name}"); plt.grid(True)
    plt.subplot(122); plt.plot(fps_list, color='blue'); plt.title(f"FPS: {algo_name}"); plt.grid(True)
    plt.tight_layout()
    plot_path = f"plot_{algo_name}.png"
    plt.savefig(plot_path)
    plt.close()

    return out_path, df_result, plot_path, f"Hoàn thành! Đã xử lý toàn bộ {processed_count} khung hình."

# --- GIAO DIỆN ---
with gr.Blocks(title="Tracking Benchmark XLA") as demo:
    history_state = gr.State([]) 

    gr.Markdown("# 🎯 Phân tích Object Tracking (Thời lượng khớp Video gốc)")
    
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="1. Tải Video")
            btn_frame = gr.Button("🖼️ Hiện khung hình để chọn vật thể")
            roi_selector = gr.Image(label="2. Quét chuột chọn vật thể", tool="select", type="numpy")
            algo = gr.Radio(["KCF", "CSRT", "MOSSE"], value="KCF", label="3. Chọn thuật toán")
            run_btn = gr.Button("🚀 CHẠY PHÂN TÍCH", variant="primary")

        with gr.Column(scale=1):
            video_output = gr.Video(label="Video kết quả")
            
            with gr.Tabs():
                with gr.Tab("📋 Bảng Thống kê"):
                    table_output = gr.Dataframe(headers=["Thuật toán", "FPS TB", "IOU TB", "Tỉ lệ Thành công"], label="Kết quả cộng dồn")
                
                with gr.Tab("📊 Biểu đồ"):
                    plot_output = gr.Image(label="Biến thiên IOU và FPS")
            
            status = gr.Textbox(label="Trạng thái")

    video_input.change(fn=on_video_change, outputs=[roi_selector, history_state, table_output, plot_output])
    btn_frame.click(fn=get_first_frame, inputs=video_input, outputs=roi_selector)
    
    run_btn.click(
        fn=run_tracking,
        inputs=[video_input, roi_selector, algo, history_state],
        outputs=[video_output, table_output, plot_output, status]
    )

if __name__ == "__main__":
    demo.launch()