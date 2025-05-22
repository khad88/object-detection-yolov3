"""
Chương trình nhận diện đối tượng từ video sử dụng YOLOv3
"""
import argparse
import time
import cv2
from config import CONFIG_PATH, WEIGHTS_PATH, DEFAULT_CONFIDENCE, DEFAULT_THRESHOLD
from detection_utils import load_yolo_model, detect_objects_yolo, draw_predictions

def main():
    # Xử lý tham số dòng lệnh
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
        help="path to input video")
    ap.add_argument("-o", "--output", required=True,
        help="path to output video")
    ap.add_argument("-c", "--confidence", type=float, default=DEFAULT_CONFIDENCE,
        help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help="threshold when applying non-maxima suppression")
    ap.add_argument("-f", "--fps", type=int, default=30,
        help="FPS of output video")
    ap.add_argument("-s", "--skip-frames", type=int, default=0,
        help="number of frames to skip between detections (to speed up processing)")
    args = vars(ap.parse_args())

    # Tải model
    net, ln = load_yolo_model(CONFIG_PATH, WEIGHTS_PATH)
    
    # Khởi tạo video capture
    print("[INFO] opening video file...")
    vs = cv2.VideoCapture(args["input"])
    
    # Lấy thông tin về số frame trong video
    try:
        total_frames = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[INFO] {total_frames} total frames in video")
    except:
        print("[INFO] could not determine # of frames in video")
        total_frames = -1
    
    # Khởi tạo các biến
    writer = None
    (W, H) = (None, None)
    frame_count = 0
    
    # Vòng lặp qua các frame trong video
    while True:
        # Đọc frame tiếp theo
        (grabbed, frame) = vs.read()
        
        # Nếu không đọc được frame, thoát khỏi vòng lặp
        if not grabbed:
            break
            
        # Lấy kích thước frame nếu chưa có
        if W is None or H is None:
            (H, W) = frame.shape[:2]
        
        # Bỏ qua frame nếu cần (để tăng tốc độ xử lý)
        frame_count += 1
        if args["skip_frames"] > 0 and frame_count % (args["skip_frames"] + 1) != 0:
            # Vẫn ghi frame này vào video output nhưng không thực hiện nhận diện
            if writer is not None:
                writer.write(frame)
            continue
        
        # Thực hiện nhận diện đối tượng
        start = time.time()
        results = detect_objects_yolo(
            net, ln, frame, 
            confidence_threshold=args["confidence"], 
            nms_threshold=args["threshold"]
        )
        end = time.time()
        
        # Vẽ kết quả nhận diện lên frame
        output_frame = draw_predictions(frame, results)
        
        # Khởi tạo video writer nếu chưa có
        if writer is None:
            # Chọn codec phù hợp
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args["output"], fourcc, args["fps"],
                (frame.shape[1], frame.shape[0]), True)
                
            # Hiển thị thông tin xử lý
            if total_frames > 0:
                elap = (end - start)
                print(f"[INFO] single frame took {elap:.4f} seconds")
                estimated_time = elap * total_frames / (args["skip_frames"] + 1)
                print(f"[INFO] estimated total time to finish: {estimated_time:.4f} seconds")
        
        # Ghi frame vào video output
        writer.write(output_frame)
        
        # Hiển thị tiến trình
        if total_frames > 0 and frame_count % 100 == 0:
            percent_complete = frame_count / total_frames * 100
            print(f"[INFO] Processing: {percent_complete:.2f}% complete")
    
    # Dọn dẹp
    print("[INFO] cleaning up...")
    if writer is not None:
        writer.release()
    vs.release()
    print(f"[INFO] Output saved to {args['output']}")

if __name__ == "__main__":
    main()