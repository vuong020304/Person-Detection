import os
import time

def calculate_fps(start_time, frame_count):
    """Calculate current FPS"""
    elapsed_time = time.time() - start_time
    return frame_count / elapsed_time if elapsed_time > 0 else 0

def validate_file_path(path):
    """Check if file exists"""
    return os.path.exists(path)

def create_output_path(input_path, suffix="_output"):
    """Create output path from input path"""
    name, ext = os.path.splitext(input_path)
    return f"{name}{suffix}{ext}"