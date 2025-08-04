import cv2
import time
from detector import PersonDetector

def real_time_person_detection(model_name="models/yolov8n.pt"):
    """
    Real-time person detection using webcam
    """
    detector = PersonDetector(model_name)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open webcam. Trying alternative camera...")
        cap = cv2.VideoCapture(1)
        
    if not cap.isOpened():
        print("❌ No webcam available.")
        return
    
    print("✓ Webcam opened successfully!")
    print("Press 'q' to quit, 's' to save frame")
    
    frame_count = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame from webcam")
            break
        
        frame_count += 1
        
        # Detect persons
        persons, _ = detector.detect_persons(frame, conf_threshold=0.25, iou_threshold=0.7)
        
        # Draw detections
        frame_with_boxes = detector.draw_detections(frame, persons)
        
        # Add processing info
        elapsed_time = time.time() - start_time
        current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        cv2.putText(frame_with_boxes, f'FPS: {current_fps:.1f}', 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_with_boxes, 'Press q:quit, s:save', 
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display
        cv2.imshow('Real-time Person Detection', frame_with_boxes)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            save_path = f"person_detection_{frame_count}.jpg"
            cv2.imwrite(save_path, frame_with_boxes)
            print(f"✓ Frame saved: {save_path}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("✓ Real-time detection completed!")