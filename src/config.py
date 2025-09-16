"""
Configuration settings for the person tracker system.
"""

import os

class Config:
    # Model settings
    YOLO_MODEL = "yolov8n.pt"  # YOLOv8 nano model - small and fast
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.45
    
    # Tracking settings
    MAX_DISAPPEARED = 30  # Max frames a person can disappear before being removed
    MAX_DISTANCE = 100    # Max distance for tracking association
    
    # Visualization settings
    BBOX_COLOR = (0, 255, 0)      # Green bounding box
    FLOW_LINE_COLOR = (0, 0, 200) # Darker red flow line for better visibility
    LINE_THICKNESS = 2
    FLOW_LINE_THICKNESS = 5       # Thicker flow lines for better visibility
    TRAIL_LENGTH = 50             # Number of points in the trail
    
    # Video settings
    DEFAULT_FPS = 30
    OUTPUT_CODEC = 'XVID'  # More compatible codec
    
    # File paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    VIDEOS_DIR = os.path.join(BASE_DIR, "videos") 
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    
    # Embedding settings
    EMBEDDING_SIMILARITY_THRESHOLD = 0.7
    REFERENCE_IMAGE_SIZE = (640, 640)
    
    # GUI settings
    ROI_WINDOW_NAME = "Select Person to Track"
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        for directory in [cls.MODELS_DIR, cls.VIDEOS_DIR, cls.OUTPUT_DIR]:
            os.makedirs(directory, exist_ok=True)
