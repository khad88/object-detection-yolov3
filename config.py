"""
Cấu hình chung cho hệ thống nhận diện đối tượng
"""
import numpy as np
import os

# Đường dẫn tới thư mục YOLO
YOLO_PATH = "yolo-coco"

# Đường dẫn tới các file của YOLO
LABELS_PATH = os.path.join(YOLO_PATH, "coco.names")
WEIGHTS_PATH = os.path.join(YOLO_PATH, "yolov3.weights")
CONFIG_PATH = os.path.join(YOLO_PATH, "yolov3.cfg")

# Đọc nhãn từ file
LABELS = open(LABELS_PATH).read().strip().split("\n")

# Thiết lập màu sắc cho các class
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# Các cài đặt mặc định
DEFAULT_CONFIDENCE = 0.5
DEFAULT_THRESHOLD = 0.3
DEFAULT_WIDTH = 416
DEFAULT_HEIGHT = 416

# MobileNet SSD classes
MOBILENET_CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]

# MobileNet SSD model paths
MOBILENET_PROTOTXT = "MobileNetSSD_deploy.prototxt.txt"
MOBILENET_MODEL = "MobileNetSSD_deploy.caffemodel"