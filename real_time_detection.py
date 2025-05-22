"""
Chương trình nhận diện đối tượng trong thời gian thực từ webcam sử dụng YOLOv3
"""
import argparse
import time
import cv2
import imutils
from imutils.video import VideoStream
from imutils.video import FPS
from config import CONFIG_PATH, WEIGHTS_PATH, DEFAULT_CONFIDENCE, DEFAULT_THRESHOLD
from detection_utils import load_yolo_model, detect_objects_yolo, draw_predictions

def main():
    # Xử lý tham số dòng lệnh
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--confidence", type=float, default=DEFAULT_CONFIDENCE,
        help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help="threshold when applying non-maxima suppression")
    ap.add_argument("-s", "--source", type=int, default=0,
        help="camera source (default is 0 for webcam)")
    ap.add_argument("-w", "--width", type=int, default=400,
        help="width of the displayed frame")
    args = vars(ap.parse_args())

    # Tải model
    net, ln = load_yolo_model(CONFIG_PATH, WEIGHTS_PATH)

    # Khởi tạo video stream
    print("[INFO] starting video stream...")
    vs = VideoStream(src=args["source"]).start()
    time.sleep(2.0)
    fps = FPS().start()

    # Vòng lặp xử lý video
    while True:
        # Lấy frame từ video stream và resize
        frame = vs.read()
        frame = imutils.resize(frame, width=args["width"])
        
        # Thực hiện nhận diện đối tượng
        results = detect_objects_yolo(
            net, ln, frame, 
            confidence_threshold=args["confidence"], 
            nms_threshold=args["threshold"]
        )
        
        # Vẽ kết quả nhận diện lên frame
        frame = draw_predictions(frame, results)
        
        # Hiển thị output frame
        cv2.imshow("Real-Time Object Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        
        # Nếu phím 'q' được nhấn, thoát khỏi vòng lặp
        if key == ord("q"):
            break
            
        # Cập nhật FPS counter
        fps.update()
    
    # Dừng timer và hiển thị thông tin FPS
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    
    # Dọn dẹp
    cv2.destroyAllWindows()
    vs.stop()

if __name__ == "__main__":
    main()