"""
Chương trình nhận diện đối tượng từ ảnh sử dụng YOLOv3
"""
import argparse
import cv2
from config import CONFIG_PATH, WEIGHTS_PATH, DEFAULT_CONFIDENCE, DEFAULT_THRESHOLD
from detection_utils import load_yolo_model, detect_objects_yolo, draw_predictions

def main():
    # Xử lý tham số dòng lệnh
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path to input image")
    ap.add_argument("-c", "--confidence", type=float, default=DEFAULT_CONFIDENCE,
        help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help="threshold when applying non-maxima suppression")
    ap.add_argument("-o", "--output", type=str,
        help="path to optional output image file")
    args = vars(ap.parse_args())

    # Tải model
    net, ln = load_yolo_model(CONFIG_PATH, WEIGHTS_PATH)

    # Đọc ảnh đầu vào
    image = cv2.imread(args["image"])
    if image is None:
        print(f"[ERROR] Could not read image from {args['image']}")
        return
    
    # Thực hiện nhận diện đối tượng
    results = detect_objects_yolo(
        net, ln, image, 
        confidence_threshold=args["confidence"], 
        nms_threshold=args["threshold"]
    )
    
    # Vẽ kết quả nhận diện lên ảnh
    output_image = draw_predictions(image.copy(), results)
    
    # In ra số lượng đối tượng được phát hiện
    print(f"[INFO] Found {len(results)} objects in the image")
    
    # Hiển thị và/hoặc lưu ảnh
    cv2.imshow("Object Detection Result", output_image)
    
    if args["output"]:
        cv2.imwrite(args["output"], output_image)
        print(f"[INFO] Output saved to {args['output']}")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()