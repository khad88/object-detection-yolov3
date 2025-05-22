"""
Module chung cho việc nhận diện đối tượng, có thể sử dụng lại giữa các file
"""
import cv2
import numpy as np
import time
from config import LABELS, COLORS, DEFAULT_CONFIDENCE, DEFAULT_THRESHOLD

def load_yolo_model(config_path, weights_path):
    """
    Tải model YOLO từ disk
    """
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    ln = net.getLayerNames()
    try:
        # OpenCV 4.5.4+
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers().flatten()]
    except:
        # Phiên bản OpenCV cũ hơn
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, ln

def load_mobilenet_model(prototxt_path, model_path):
    """
    Tải model MobileNet SSD từ disk
    """
    print("[INFO] loading MobileNet SSD model...")
    return cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

def create_yolo_blob(image, width=416, height=416):
    """
    Tạo blob từ ảnh đầu vào cho model YOLO
    """
    return cv2.dnn.blobFromImage(image, 1 / 255.0, (width, height),
        swapRB=True, crop=False)

def create_mobilenet_blob(image, width=300, height=300):
    """
    Tạo blob từ ảnh đầu vào cho model MobileNet SSD
    """
    return cv2.dnn.blobFromImage(cv2.resize(image, (width, height)),
        0.007843, (width, height), 127.5)

def detect_objects_yolo(net, ln, image, confidence_threshold=DEFAULT_CONFIDENCE, nms_threshold=DEFAULT_THRESHOLD):
    """
    Thực hiện nhận diện đối tượng bằng YOLO trên một ảnh
    """
    (H, W) = image.shape[:2]
    
    # Tạo blob và forward pass
    blob = create_yolo_blob(image)
    net.setInput(blob)
    start = time.time()
    layer_outputs = net.forward(ln)
    end = time.time()
    
    print(f"[INFO] YOLO took {end - start:.6f} seconds")
    
    # Khởi tạo danh sách kết quả
    boxes = []
    confidences = []
    classIDs = []
    
    # Xử lý kết quả từ các layer output
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if confidence > confidence_threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    
    # Áp dụng non-maxima suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    
    results = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            results.append({
                "class_id": classIDs[i],
                "label": LABELS[classIDs[i]],
                "confidence": confidences[i],
                "box": (x, y, w, h)
            })
    
    return results

def detect_objects_mobilenet(net, image, confidence_threshold=0.2):
    """
    Thực hiện nhận diện đối tượng bằng MobileNet SSD trên một ảnh
    """
    from config import MOBILENET_CLASSES
    
    (h, w) = image.shape[:2]
    blob = create_mobilenet_blob(image)
    net.setInput(blob)
    detections = net.forward()
    
    results = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > confidence_threshold:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Chuyển đổi từ (startX, startY, endX, endY) sang (x, y, w, h)
            x, y = startX, startY
            w, h = endX - startX, endY - startY
            
            results.append({
                "class_id": idx,
                "label": MOBILENET_CLASSES[idx],
                "confidence": confidence,
                "box": (x, y, w, h)
            })
    
    return results

def draw_predictions(image, results):
    """
    Vẽ kết quả nhận diện lên ảnh
    """
    for result in results:
        # Lấy thông tin từ kết quả
        label = result["label"]
        confidence = result["confidence"]
        (x, y, w, h) = result["box"]
        class_id = result["class_id"]
        
        # Vẽ hộp giới hạn và nhãn
        color = [int(c) for c in COLORS[class_id]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(label, confidence)
        y_pos = y - 5 if y - 5 > 15 else y + 15
        cv2.putText(image, text, (x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return image