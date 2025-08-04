# Model settings
MODEL_CONFIG = {
    'yolov8n': {
        'path': 'models/yolov8n.pt',
        'conf_threshold': 0.15,
        'iou_threshold': 0.7,
        'max_det': 50
    },
    'yolov11s': {
        'path': 'models/yolov11s.pt',
        'conf_threshold': 0.25,
        'iou_threshold': 0.7,
        'max_det': 100
    }
}

# Display settings
DISPLAY_CONFIG = {
    'show_confidence': True,
    'show_count': True,
    'show_fps': True,
    'box_thickness': 2,
    'text_scale': 0.6
}