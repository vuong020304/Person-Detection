import os
import cv2
from detector import PersonDetector
from video_processor import process_video
from real_time import real_time_person_detection
from utils import validate_file_path

if __name__ == "__main__":
    print("=== YOLOv11 Person Detection ===")
    print("1. Real-time detection with webcam")
    print("2. Process video file")
    print("3. Process image file")
    
    choice = input("Choose option (1/2/3): ").strip()
    
    if choice == "1":
        # Real-time detection
        real_time_person_detection()
        
    elif choice == "2":
        # Video processing
        video_path = input("Enter video path: ").strip().strip('"').strip("'")
        output_path = input("Enter output video path (optional): ").strip().strip('"').strip("'")
        
        if validate_file_path(video_path):
            if output_path == "":
                output_path = None
            process_video(video_path, output_path)
        else:
            print(f"❌ Video file not found: {video_path}")
            
    elif choice == "3":
        # Image processing
        image_path = input("Enter image path: ").strip().strip('"').strip("'")
        
        if validate_file_path(image_path):
            detector = PersonDetector()
            persons, image = detector.detect_persons(image_path, conf_threshold=0.25, iou_threshold=0.7)
            result = detector.draw_detections(image, persons)
            
            cv2.imshow('Person Detection Result', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            print(f"✓ Detected {len(persons)} persons")
        else:
            print(f"❌ Image file not found: {image_path}")
            
    else:
        print("Invalid choice. Running real-time detection...")
        real_time_person_detection()