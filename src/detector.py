from ultralytics import YOLO
import cv2
import numpy as np

class PersonDetector:
    
    "Phát hiện người trên ảnh/ video với nhiều tính năng nâng cao"

    "Khởi tạo model YOLO"

    def __init__(self, model_name="models/yolov8n.pt"):
        print(f"Loading YOLO model: {model_name}")
        self.model = YOLO(model_name)
        print("✓ Model loaded successfully!")
    
    "Nhận diện person"
    def detect_persons(self, image, conf_threshold=0.25, iou_threshold=0.7):
        if isinstance(image, str):
            image = cv2.imread(image)

        "Chạy YOLO detection với tham số tối ưu"
        results = self.model(image, 
                           conf=conf_threshold,
                           iou=iou_threshold,
                           agnostic_nms=True,
                           max_det=50)
        
        "Lấy kết quả"
        persons = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    "Lấy tọa độ và độ tin cậy"
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())

                    "Lọc person class (class 0 trong COCO dataset)"
                    if class_id == 0:
                        persons.append({
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": float(confidence)
                        })
        return persons, image
    
    "Vẽ bounding box và độ tin cậy"
    def draw_detections(self, image, persons, show_confidence=True, show_count=True):
        image_with_boxes = image.copy()

        for i, person in enumerate(persons):
            x1, y1, x2, y2 = person['bbox']
            confidence = person['confidence']

            # Tính toán điểm giữa cho ID person
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # Vẽ bounding box với màu dựa trên độ tin cậy
            if confidence > 0.8:
                color = (0, 255, 0) # Green for high confidence
            elif confidence > 0.6:
                color = (0, 255, 255) # Yellow for medium confidence
            else:
                color = (0, 165, 255) # Orange for low confidence

            # Vẽ bounding box
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 2)

            # Vẽ ID person
            cv2.putText(image_with_boxes, f'Person {i+1}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Vẽ độ tin cậy
            if show_confidence:
                cv2.putText(image_with_boxes, f'{confidence:.2f}', (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Vẽ tổng số person
        if show_count:
            cv2.putText(image_with_boxes, f'Total Persons: {len(persons)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return image_with_boxes 