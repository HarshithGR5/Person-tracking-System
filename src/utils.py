"""
Utility functions for video processing, file handling, and data management.
"""

import cv2
import json
import csv
import os
from typing import Tuple, List, Dict, Optional, Any
import numpy as np
from datetime import datetime
from config import Config


def get_video_info(video_path: str) -> Dict[str, Any]:
    """
    Get video information including FPS, frame count, resolution, etc.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video information
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration': 0.0
    }
    
    # Calculate duration
    if info['fps'] > 0:
        info['duration'] = info['frame_count'] / info['fps']
    
    cap.release()
    return info


def frame_to_timestamp(frame_number: int, fps: float) -> str:
    """
    Convert frame number to timestamp string.
    
    Args:
        frame_number: Frame number
        fps: Video FPS
        
    Returns:
        Timestamp string in format "MM:SS.mmm"
    """
    if fps <= 0:
        return "00:00.000"
    
    total_seconds = frame_number / fps
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    
    return f"{minutes:02d}:{seconds:06.3f}"


def timestamp_to_frame(timestamp_str: str, fps: float) -> int:
    """
    Convert timestamp string to frame number.
    
    Args:
        timestamp_str: Timestamp in format "MM:SS.mmm" or "SS.mmm"
        fps: Video FPS
        
    Returns:
        Frame number
    """
    if fps <= 0:
        return 0
    
    # Parse timestamp
    parts = timestamp_str.split(':')
    if len(parts) == 2:
        minutes = int(parts[0])
        seconds = float(parts[1])
    else:
        minutes = 0
        seconds = float(parts[0])
    
    total_seconds = minutes * 60 + seconds
    return int(total_seconds * fps)


def save_tracking_results(output_path: str, results: Dict[str, Any]) -> None:
    """
    Save tracking results to JSON file.
    
    Args:
        output_path: Path to save results
        results: Dictionary with tracking results
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Add metadata
    results['metadata'] = {
        'created_at': datetime.now().isoformat(),
        'version': '1.0'
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def save_trajectory_csv(output_path: str, trajectory: List[Tuple[int, int, int]], 
                       fps: float, track_id: Optional[int] = None) -> None:
    """
    Save trajectory data to CSV file.
    
    Args:
        output_path: Path to save CSV
        trajectory: List of (x, y, frame_number) tuples
        fps: Video FPS
        track_id: Optional track ID
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        headers = ['frame_number', 'timestamp', 'x', 'y']
        if track_id is not None:
            headers.insert(0, 'track_id')
        writer.writerow(headers)
        
        # Write data
        for x, y, frame_num in trajectory:
            timestamp = frame_to_timestamp(frame_num, fps)
            row = [frame_num, timestamp, x, y]
            if track_id is not None:
                row.insert(0, track_id)
            writer.writerow(row)


def load_trajectory_csv(csv_path: str) -> List[Tuple[int, int, int]]:
    """
    Load trajectory data from CSV file.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        List of (x, y, frame_number) tuples
    """
    trajectory = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = int(row['x'])
            y = int(row['y'])
            frame_num = int(row['frame_number'])
            trajectory.append((x, y, frame_num))
    
    return trajectory


def create_video_writer(output_path: str, width: int, height: int, fps: float) -> cv2.VideoWriter:
    """
    Create video writer for output video.
    
    Args:
        output_path: Output video path
        width: Video width
        height: Video height
        fps: Video FPS
        
    Returns:
        OpenCV VideoWriter object
    """
    # Ensure output directory exists (only if there's a directory path)
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if path contains one
        os.makedirs(output_dir, exist_ok=True)
    
    # Try different codecs in order of preference
    # Keep MP4 format for MP4 inputs to avoid conversion issues
    codecs_to_try = [
        ('mp4v', '.mp4'),    # MP4 native - best for MP4 inputs
        ('H264', '.mp4'),    # Modern H.264
        ('MJPG', '.avi'),    # Motion JPEG fallback
        ('XVID', '.avi'),    # AVI fallback
        ('DIVX', '.avi'),    # Alternative AVI codec
        ('avc1', '.mp4')     # Another H.264 variant
    ]
    
    for codec, ext in codecs_to_try:
        # Adjust output path extension if needed
        base_path = os.path.splitext(output_path)[0]
        test_path = base_path + ext
        
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(test_path, fourcc, fps, (width, height))
            
            if writer.isOpened():
                print(f"✓ Video writer created with {codec} codec: {test_path}")
                return writer
            else:
                writer.release()
        except Exception as e:
            print(f"Failed to create writer with {codec}: {e}")
            continue
    
    # Fallback - try without specifying codec
    try:
        fourcc = 0  # Let OpenCV choose
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if writer.isOpened():
            print(f"✓ Video writer created with default codec: {output_path}")
            return writer
    except Exception as e:
        print(f"Fallback writer creation failed: {e}")
    
    raise ValueError(f"Could not create video writer with any codec for: {output_path}")


def validate_video_file(video_path: str) -> bool:
    """
    Validate if video file can be opened and read.
    
    Args:
        video_path: Path to video file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        return ret and frame is not None
    except:
        return False


def resize_frame(frame: np.ndarray, max_width: int = 1920, max_height: int = 1080) -> np.ndarray:
    """
    Resize frame while maintaining aspect ratio.
    
    Args:
        frame: Input frame
        max_width: Maximum width
        max_height: Maximum height
        
    Returns:
        Resized frame
    """
    h, w = frame.shape[:2]
    
    # Calculate scaling factor
    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h, 1.0)  # Don't upscale
    
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))
    
    return frame


def calculate_bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    Calculate center point of bounding box.
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2)
        
    Returns:
        Center point (x, y)
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return center_x, center_y


def calculate_bbox_area(bbox: Tuple[int, int, int, int]) -> int:
    """
    Calculate area of bounding box.
    
    Args:
        bbox: Bounding box (x1, y1, x2, y2)
        
    Returns:
        Area in pixels
    """
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)


def calculate_distance(point1: Tuple[int, int], point2: Tuple[int, int]) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: First point (x, y)
        point2: Second point (x, y)
        
    Returns:
        Distance in pixels
    """
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def smooth_trajectory(trajectory: List[Tuple[int, int]], window_size: int = 5) -> List[Tuple[int, int]]:
    """
    Smooth trajectory using moving average.
    
    Args:
        trajectory: List of (x, y) points
        window_size: Size of smoothing window
        
    Returns:
        Smoothed trajectory
    """
    if len(trajectory) <= window_size:
        return trajectory
    
    smoothed = []
    half_window = window_size // 2
    
    for i in range(len(trajectory)):
        start = max(0, i - half_window)
        end = min(len(trajectory), i + half_window + 1)
        
        # Calculate average
        x_sum = sum(x for x, y in trajectory[start:end])
        y_sum = sum(y for x, y in trajectory[start:end])
        count = end - start
        
        smoothed.append((x_sum // count, y_sum // count))
    
    return smoothed


def create_output_filename(base_name: str, suffix: str = "", extension: str = ".mp4") -> str:
    """
    Create output filename with timestamp.
    
    Args:
        base_name: Base name for file
        suffix: Optional suffix
        extension: File extension
        
    Returns:
        Generated filename
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if suffix:
        filename = f"{base_name}_{suffix}_{timestamp}{extension}"
    else:
        filename = f"{base_name}_{timestamp}{extension}"
    
    return filename


def ensure_directory(directory_path: str) -> None:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory_path: Path to directory
    """
    os.makedirs(directory_path, exist_ok=True)


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in megabytes.
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in MB
    """
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except:
        return 0.0


def progress_callback(current: int, total: int, message: str = "Processing") -> None:
    """
    Simple progress callback for long operations.
    
    Args:
        current: Current progress
        total: Total items
        message: Progress message
    """
    if total > 0:
        percentage = (current / total) * 100
        print(f"\r{message}: {percentage:.1f}% ({current}/{total})", end='', flush=True)
        
        if current == total:
            print()  # New line when complete
