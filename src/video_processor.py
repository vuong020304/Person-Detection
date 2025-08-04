import cv2
import time
import os
from detector import PersonDetector

def process_video(video_path, output_path=None, model_name="models/yolov8n.pt"):
    """
    Process video file for person detection
    """
    detector = PersonDetector(model_name)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video Info: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup video writer if output path is provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    start_time = time.time()
    
    print("Processing video...")
    print("Press 'q' to quit, 's' to save frame")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect persons
        persons, _ = detector.detect_persons(frame, conf_threshold=0.25, iou_threshold=0.7)
                
        # Draw detections
        frame_with_boxes = detector.draw_detections(frame, persons)
        
        # Add processing info
        elapsed_time = time.time() - start_time
        current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        cv2.putText(frame_with_boxes, f'Frame: {frame_count}/{total_frames}', 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_with_boxes, f'FPS: {current_fps:.1f}', 
                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame_with_boxes, 'Press q:quit, s:save', 
                   (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display
        cv2.imshow('Person Detection - Video', frame_with_boxes)
        
        # Write to output video if specified
        if writer:
            writer.write(frame_with_boxes)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            save_path = f"frame_{frame_count}.jpg"
            cv2.imwrite(save_path, frame_with_boxes)
            print(f"✓ Frame saved: {save_path}")
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    print(f"✓ Video processing completed!")
    print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds")
    print(f"Average FPS: {frame_count/elapsed_time:.1f}")