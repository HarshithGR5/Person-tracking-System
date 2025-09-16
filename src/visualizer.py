"""
Visualization module for drawing flow lines, bounding boxes, and trajectory overlays.
Handles different view modes and visual effects.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from config import Config


class Visualizer:
    def __init__(self):
        """Initialize the visualizer with default settings."""
        self.bbox_color = Config.BBOX_COLOR
        self.flow_line_color = Config.FLOW_LINE_COLOR
        self.line_thickness = Config.LINE_THICKNESS
        self.flow_line_thickness = Config.FLOW_LINE_THICKNESS
        
        # Colors for different tracks (cycling through colors)
        self.track_colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
            (0, 128, 255),  # Light Blue
            (128, 255, 0),  # Light Green
        ]
        
    def get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """
        Get a consistent color for a track ID.
        
        Args:
            track_id: Track ID
            
        Returns:
            BGR color tuple
        """
        try:
            color_index = int(track_id) % len(self.track_colors)
            return self.track_colors[color_index]
        except Exception as e:
            # Return default color if there's any issue
            return (255, 0, 0)  # Red as default
    
    def draw_bounding_box(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                         track_id: Optional[int] = None, confidence: Optional[float] = None,
                         is_target: bool = False) -> np.ndarray:
        """
        Draw bounding box on frame.
        
        Args:
            frame: Input frame
            bbox: Bounding box (x1, y1, x2, y2)
            track_id: Optional track ID
            confidence: Optional confidence score
            is_target: Whether this is the target person
            
        Returns:
            Frame with bounding box drawn
        """
        x1, y1, x2, y2 = bbox
        
        # Choose color and style based on whether it's the target
        if is_target:
            # Make target bounding box much more prominent
            color = (0, 255, 0)  # Bright green for target
            thickness = max(4, self.line_thickness + 3)  # Much thicker line
            # Draw double border for extra visibility
            cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), (0, 0, 0), thickness+2)  # Black outer border
        else:
            # Don't draw bounding boxes for non-target people
            return frame
        
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Prepare label text
        label_parts = []
        if track_id is not None:
            label_parts.append(f"ID: {track_id}")
        if confidence is not None:
            label_parts.append(f"{confidence:.2f}")
        if is_target:
            label_parts.append("🎯 TARGET")
        
        if label_parts:
            label = " | ".join(label_parts)
            
            # Calculate text size - make target labels larger
            font = cv2.FONT_HERSHEY_SIMPLEX
            if is_target:
                font_scale = 0.8  # Larger font for target
                font_thickness = 3
                text_color = (255, 255, 255)  # White text
                bg_color = (0, 255, 0)  # Green background
            else:
                font_scale = 0.6
                font_thickness = 2
                text_color = (255, 255, 255)
                bg_color = color
            
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # Draw enhanced label background
            bg_rect_height = text_height + 15 if is_target else text_height + 10
            cv2.rectangle(frame, (x1, y1 - bg_rect_height), (x1 + text_width + 10, y1), bg_color, -1)
            
            # Add border for target labels
            if is_target:
                cv2.rectangle(frame, (x1, y1 - bg_rect_height), (x1 + text_width + 10, y1), (0, 0, 0), 2)
            
            # Draw label text
            cv2.putText(frame, label, (x1 + 5, y1 - 8), font, font_scale, text_color, font_thickness)
        
        return frame
    
    def draw_flow_line(self, frame: np.ndarray, trajectory: List[Tuple[int, int]], 
                      color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        """
        Draw enhanced, highly visible flow line (trajectory) on frame.
        
        Args:
            frame: Input frame
            trajectory: List of (x, y) points
            color: Optional color override
            
        Returns:
            Frame with flow line drawn
        """
        if len(trajectory) < 2:
            return frame
        
        # Use bright, highly visible colors
        line_color = color or (0, 255, 255)  # Bright yellow/cyan
        shadow_color = (0, 0, 0)  # Black shadow for contrast
        
        # Convert trajectory to numpy array
        points = np.array(trajectory, dtype=np.int32)
        
        # Draw shadow/outline for better visibility
        thick_line_thickness = max(8, self.flow_line_thickness + 4)
        cv2.polylines(frame, [points], False, shadow_color, thick_line_thickness)
        
        # Draw main trajectory line (thicker and brighter)
        main_line_thickness = max(6, self.flow_line_thickness + 2)
        cv2.polylines(frame, [points], False, line_color, main_line_thickness)
        
        # Draw gradient line from start to end for direction indication
        if len(trajectory) > 1:
            # Create color gradient from blue to red (showing direction)
            for i in range(len(trajectory) - 1):
                progress = i / (len(trajectory) - 1)
                # Color transition: Blue -> Green -> Yellow -> Red
                if progress < 0.33:
                    # Blue to Green
                    local_progress = progress / 0.33
                    gradient_color = (255, int(255 * local_progress), 0)
                elif progress < 0.66:
                    # Green to Yellow
                    local_progress = (progress - 0.33) / 0.33
                    gradient_color = (255, 255, int(255 * local_progress))
                else:
                    # Yellow to Red
                    local_progress = (progress - 0.66) / 0.34
                    gradient_color = (int(255 * (1 - local_progress)), 255, 255)
                
                pt1 = trajectory[i]
                pt2 = trajectory[i + 1]
                cv2.line(frame, pt1, pt2, gradient_color, 4)
        
        # Draw enhanced points with pulsing effect
        for i, (x, y) in enumerate(trajectory):
            # Calculate size for pulsing effect (newer points are larger)
            progress = (i + 1) / len(trajectory)
            base_radius = 3
            pulse_radius = int(base_radius + 4 * progress)
            
            # Draw outer glow
            cv2.circle(frame, (x, y), pulse_radius + 2, (255, 255, 255), 2)
            # Draw main point
            cv2.circle(frame, (x, y), pulse_radius, line_color, -1)
            # Draw inner highlight
            cv2.circle(frame, (x, y), max(1, pulse_radius - 2), (255, 255, 255), -1)
        
        # Highlight the most recent point with special animation
        if trajectory:
            x, y = trajectory[-1]
            # Large pulsing circle for current position
            cv2.circle(frame, (x, y), 12, (0, 0, 0), 3)  # Black border
            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)  # Bright green fill
            cv2.circle(frame, (x, y), 6, (255, 255, 255), -1)  # White center
            
            # Add direction arrow if we have enough points
            if len(trajectory) >= 2:
                prev_x, prev_y = trajectory[-2]
                # Calculate direction vector
                dx = x - prev_x
                dy = y - prev_y
                length = np.sqrt(dx*dx + dy*dy)
                if length > 5:  # Only draw arrow if there's significant movement
                    # Normalize and scale
                    dx, dy = dx/length, dy/length
                    arrow_length = 20
                    arrow_x = int(x + dx * arrow_length)
                    arrow_y = int(y + dy * arrow_length)
                    
                    # Draw direction arrow
                    cv2.arrowedLine(frame, (x, y), (arrow_x, arrow_y), 
                                  (0, 255, 0), 3, tipLength=0.3)
        
        return frame
    
    def draw_trajectory_with_fade(self, frame: np.ndarray, trajectory: List[Tuple[int, int]], 
                                 color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        """
        Draw trajectory with fading effect from old to new points.
        
        Args:
            frame: Input frame
            trajectory: List of (x, y) points
            color: Optional color override
            
        Returns:
            Frame with faded trajectory drawn
        """
        if len(trajectory) < 2:
            return frame
        
        line_color = color or self.flow_line_color
        
        # Draw line segments with varying thickness and opacity
        for i in range(len(trajectory) - 1):
            x1, y1 = trajectory[i]
            x2, y2 = trajectory[i + 1]
            
            # Calculate fade factor
            fade_factor = (i + 1) / len(trajectory)
            
            # Adjust color intensity
            faded_color = tuple(int(c * fade_factor) for c in line_color)
            
            # Adjust thickness
            thickness = max(1, int(self.flow_line_thickness * fade_factor))
            
            # Draw line segment
            cv2.line(frame, (x1, y1), (x2, y2), faded_color, thickness)
        
        return frame
    
    def draw_all_tracks(self, frame: np.ndarray, tracked_objects: List[Tuple[int, int, int, int, int]],
                       trajectories: Dict[int, List[Tuple[int, int]]] = None,
                       target_person_id: Optional[int] = None,
                       view_mode: str = "flow",
                       complete_target_trajectory: List[Tuple[int, int]] = None) -> np.ndarray:
        """
        Draw only target person's track and trajectory (others are filtered out).
        
        Args:
            frame: Input frame
            tracked_objects: List of (x1, y1, x2, y2, track_id) tuples
            trajectories: Dictionary of track_id -> trajectory points
            target_person_id: ID of target person to highlight
            view_mode: "box" for bounding boxes only, "flow" for boxes + trajectories
            complete_target_trajectory: Complete trajectory for target person (optional)
            
        Returns:
            Frame with target person visualization only
        """
        result_frame = frame.copy()
        
        # Only draw trajectory for target person
        if view_mode in ["flow", "trail"] and target_person_id is not None:
            # Use complete trajectory if provided, otherwise fall back to regular trajectory
            trajectory_to_draw = None
            
            if complete_target_trajectory and len(complete_target_trajectory) > 1:
                trajectory_to_draw = complete_target_trajectory
            elif trajectories and target_person_id in trajectories:
                trajectory_to_draw = trajectories[target_person_id]
            
            if trajectory_to_draw and len(trajectory_to_draw) > 1:
                color = (0, 0, 255)  # Red color for target
                
                if view_mode == "flow":
                    # Use fading effect for target trajectory
                    result_frame = self.draw_trajectory_with_fade(result_frame, trajectory_to_draw, color)
                else:  # trail mode
                    # Simple line for trail view
                    result_frame = self.draw_flow_line(result_frame, trajectory_to_draw, color)
        
        # Only draw bounding box for target person
        for x1, y1, x2, y2, track_id in tracked_objects:
            if track_id == target_person_id:  # Only draw target person's box
                result_frame = self.draw_bounding_box(
                    result_frame, (x1, y1, x2, y2), track_id, is_target=True
                )
        
        return result_frame
    
    def add_info_overlay(self, frame: np.ndarray, frame_number: int, fps: float,
                        target_info: Dict, total_frames: Optional[int] = None) -> np.ndarray:
        """
        Add information overlay to the frame.
        
        Args:
            frame: Input frame
            frame_number: Current frame number
            fps: Video FPS
            target_info: Target person information
            total_frames: Total frames in video (optional)
            
        Returns:
            Frame with info overlay
        """
        # Calculate timestamp
        timestamp = frame_number / fps if fps > 0 else 0
        minutes = int(timestamp // 60)
        seconds = timestamp % 60
        
        # Prepare info text
        info_lines = [
            f"Frame: {frame_number}" + (f"/{total_frames}" if total_frames else ""),
            f"Time: {minutes:02d}:{seconds:05.2f}",
            f"FPS: {fps:.1f}"
        ]
        
        # Add target info
        if target_info['target_id'] is not None:
            info_lines.extend([
                f"Target ID: {target_info['target_id']}",
                f"Trajectory Length: {target_info['trajectory_length']}",
                f"Status: {'Active' if target_info['is_active'] else 'Lost'}"
            ])
            
            # Add validation info if available
            if 'current_confidence' in target_info and target_info['current_confidence'] > 0:
                info_lines.append(f"Confidence: {target_info['current_confidence']:.3f}")
                if 'recent_avg_confidence' in target_info and target_info['recent_avg_confidence'] > 0:
                    info_lines.append(f"Avg (5f): {target_info['recent_avg_confidence']:.3f}")
            
            # Add first seen info with more prominent display
            if target_info['first_seen_frame'] is not None:
                first_seen_time = target_info['first_seen_frame'] / fps if fps > 0 else 0
                first_min = int(first_seen_time // 60)
                first_sec = first_seen_time % 60
                info_lines.append(f"⏰ First Seen: {first_min:02d}:{first_sec:05.2f}")
                
                # Show time since first appearance
                current_time = frame_number / fps if fps > 0 else 0
                time_tracked = current_time - first_seen_time
                track_min = int(time_tracked // 60)
                track_sec = time_tracked % 60
                info_lines.append(f"📊 Tracking: {track_min:02d}:{track_sec:05.2f}")
        else:
            info_lines.append("Target: Not Set")
        
        # Draw semi-transparent background
        overlay = frame.copy()
        overlay_height = len(info_lines) * 25 + 20
        cv2.rectangle(overlay, (10, 10), (400, overlay_height), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)
        
        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        
        for i, line in enumerate(info_lines):
            y_pos = 35 + i * 25
            cv2.putText(frame, line, (20, y_pos), font, font_scale, (255, 255, 255), font_thickness)
        
        return frame
    
    def draw_roi_selection(self, frame: np.ndarray, start_point: Optional[Tuple[int, int]] = None,
                          end_point: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Draw ROI selection rectangle during manual selection.
        
        Args:
            frame: Input frame
            start_point: Start point of selection
            end_point: End point of selection
            
        Returns:
            Frame with ROI rectangle
        """
        if start_point is None or end_point is None:
            return frame
        
        result_frame = frame.copy()
        
        # Draw selection rectangle
        cv2.rectangle(result_frame, start_point, end_point, (0, 255, 255), 2)
        
        # Add instruction text
        cv2.putText(result_frame, "Release to confirm selection", 
                   (start_point[0], start_point[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return result_frame
    
    def create_side_by_side_view(self, frame1: np.ndarray, frame2: np.ndarray,
                                labels: Tuple[str, str] = ("Box View", "Flow View")) -> np.ndarray:
        """
        Create side-by-side comparison of two view modes.
        
        Args:
            frame1: First frame (e.g., box view)
            frame2: Second frame (e.g., flow view)
            labels: Labels for each frame
            
        Returns:
            Combined side-by-side frame
        """
        # Ensure both frames have the same height
        h1, w1 = frame1.shape[:2]
        h2, w2 = frame2.shape[:2]
        
        if h1 != h2:
            # Resize to match the smaller height
            target_height = min(h1, h2)
            frame1 = cv2.resize(frame1, (int(w1 * target_height / h1), target_height))
            frame2 = cv2.resize(frame2, (int(w2 * target_height / h2), target_height))
        
        # Combine horizontally
        combined = np.hstack([frame1, frame2])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_thickness = 2
        
        # Label for left frame
        cv2.putText(combined, labels[0], (10, 30), font, font_scale, (255, 255, 255), font_thickness)
        
        # Label for right frame
        cv2.putText(combined, labels[1], (frame1.shape[1] + 10, 30), 
                   font, font_scale, (255, 255, 255), font_thickness)
        
        return combined
    
    def add_target_detected_notification(self, frame: np.ndarray, target_id: int, 
                                        detection_time: str, frames_since_detection: int) -> np.ndarray:
        """
        Add a special notification when target person is first detected.
        
        Args:
            frame: Input frame
            target_id: Target person ID
            detection_time: First detection timestamp
            frames_since_detection: Number of frames since first detection
            
        Returns:
            Frame with notification overlay
        """
        # Show notification for first 150 frames (about 5 seconds at 30fps)
        if frames_since_detection > 150:
            return frame
            
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # Notification box dimensions
        box_width = min(w - 40, 600)
        box_height = 120
        box_x = (w - box_width) // 2
        box_y = 50
        
        # Draw notification box with gradient effect
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), 
                     (50, 150, 50), -1)  # Green background
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), 
                     (0, 255, 0), 3)  # Green border
        
        # Add text
        text_lines = [
            "🎯 TARGET PERSON DETECTED!",
            f"ID: {target_id}  |  First seen at: {detection_time}"
        ]
        
        # Calculate text positions
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        for i, text in enumerate(text_lines):
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = box_x + (box_width - text_size[0]) // 2
            text_y = box_y + 40 + i * 35
            
            # Add text shadow
            cv2.putText(overlay, text, (text_x + 2, text_y + 2), font, font_scale, (0, 0, 0), thickness + 1)
            # Add main text
            cv2.putText(overlay, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        
        # Blend overlay with original frame
        alpha = 0.8  # Transparency
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
