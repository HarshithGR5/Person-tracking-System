"""
Main application for person tracking with flow line visualization.
Handles command-line arguments, video processing, and output generation.
"""

import argparse
import cv2
import os
import sys
import warnings
import numpy as np
from typing import Optional, List, Tuple

# Suppress pkg_resources and other deprecation warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import tkinter as tk
from tkinter import filedialog
import json

# Import our modules
from config import Config
from detector import PersonDetector
from tracker import PersonTracker
from visualizer import Visualizer
from utils import (
    get_video_info, frame_to_timestamp, save_tracking_results,
    save_trajectory_csv, create_video_writer, validate_video_file,
    resize_frame, progress_callback, create_output_filename
)


class PersonTrackingApp:
    def __init__(self, args):
        """Initialize the person tracking application."""
        self.args = args
        
        # Create necessary directories
        Config.create_directories()
        
        # Initialize components
        self.detector = PersonDetector(confidence_threshold=args.confidence)
        self.tracker = PersonTracker()
        self.visualizer = Visualizer()
        
        # Video properties
        self.video_info = None
        self.reference_embedding = None
        
        # Target detection tracking
        self.target_first_detection_frame = None
        self.target_first_detection_time = None
        
        # Output settings
        self.view_mode = args.view
        
    def load_reference_image(self, ref_image_path: str) -> bool:
        """
        Load and process reference image for target person identification.
        
        Args:
            ref_image_path: Path to reference image
            
        Returns:
            True if successful, False otherwise
        """
        print(f"Loading reference image: {ref_image_path}")
        
        # Load reference image
        ref_image = self.detector.load_reference_image(ref_image_path)
        if ref_image is None:
            print("Failed to load reference image")
            return False
        
        # Extract embedding
        self.reference_embedding = self.detector.compute_embedding(ref_image)
        
        if np.linalg.norm(self.reference_embedding) == 0:
            print("Failed to extract embedding from reference image")
            return False
        
        print("Reference image loaded successfully")
        return True
    
    def manual_roi_selection(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Allow user to manually select ROI for target person.
        
        Args:
            frame: Current frame
            
        Returns:
            Selected ROI (x1, y1, x2, y2) or None if cancelled
        """
        print("Please select the person to track...")
        
        # Use OpenCV's selectROI function
        roi = cv2.selectROI(Config.ROI_WINDOW_NAME, frame, False, False)
        cv2.destroyWindow(Config.ROI_WINDOW_NAME)
        
        if roi[2] > 0 and roi[3] > 0:  # Width and height > 0
            x, y, w, h = roi
            return (x, y, x + w, y + h)
        
        return None
    
    def process_video(self, video_path: str, output_path: str) -> bool:
        """
        Process video and generate tracked output.
        
        Args:
            video_path: Input video path
            output_path: Output video path
            
        Returns:
            True if successful, False otherwise
        """
        # Validate input video
        if not validate_video_file(video_path):
            print(f"Invalid video file: {video_path}")
            return False
        
        # Get video information
        self.video_info = get_video_info(video_path)
        print(f"Video info: {self.video_info}")
        
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        
        # Create video writer with reasonable resolution
        writer = None
        output_width, output_height = self.video_info['width'], self.video_info['height']
        
        # Scale down if resolution is too high for codec compatibility
        if output_width > 1920 or output_height > 1080:
            # Scale down to 1080p while maintaining aspect ratio
            aspect_ratio = output_width / output_height
            if aspect_ratio > 16/9:  # Wide aspect ratio
                output_width = 1920
                output_height = int(1920 / aspect_ratio)
            else:
                output_height = 1080
                output_width = int(1080 * aspect_ratio)
            print(f"Scaling output resolution from {self.video_info['width']}x{self.video_info['height']} to {output_width}x{output_height}")
        
        if output_path:
            # Match output format to input format to avoid conversion issues
            input_ext = os.path.splitext(video_path)[1].lower()
            output_ext = os.path.splitext(output_path)[1].lower()
            
            if input_ext != output_ext:
                # Change output extension to match input
                output_base = os.path.splitext(output_path)[0]
                output_path = output_base + input_ext
                print(f"Matching output format to input: {output_ext} -> {input_ext}")
                print(f"Output file: {output_path}")
            
            writer = create_video_writer(
                output_path, 
                output_width,
                output_height,
                self.video_info['fps']
            )
        
        # Processing variables
        frame_number = 0
        target_found = False
        manual_selection_done = False
        
        # Results storage
        results = {
            'video_path': video_path,
            'output_path': output_path,
            'video_info': self.video_info,
            'target_person': {
                'first_seen_frame': None,
                'first_seen_timestamp': None,
                'track_id': None
            },
            'processing_info': {
                'total_frames_processed': 0,
                'view_mode': self.view_mode
            }
        }
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_number += 1
                
                # Resize frame if too large
                display_frame = resize_frame(frame)
                
                # Detect persons in frame
                detections = self.detector.detect_persons(display_frame)
                
                # Update tracker
                tracked_objects = self.tracker.update(detections, display_frame)
                
                # Handle target person identification
                if not target_found:
                    if self.reference_embedding is not None:
                        # Try to find target using reference embedding
                        print(f"Frame {frame_number}: Looking for target person using reference image...")
                        embeddings = []
                        for det in detections:
                            crop = self.detector.extract_person_crop(display_frame, det[:4])
                            embedding = self.detector.compute_embedding(crop)
                            embeddings.append(embedding)
                        
                        target_detection_idx = self.tracker.find_target_person_from_embedding(
                            detections, embeddings, self.reference_embedding, None, frame, self.detector
                        )
                        
                        if target_detection_idx is not None:
                            # Find the track ID for this detection
                            if target_detection_idx < len(tracked_objects):
                                target_track_id = tracked_objects[target_detection_idx][4]  # track_id is 5th element
                                target_found = True
                                results['target_person']['track_id'] = target_track_id
                                results['target_person']['first_seen_frame'] = frame_number
                                results['target_person']['first_seen_timestamp'] = frame_to_timestamp(
                                    frame_number, self.video_info['fps']
                                )
                                # Calculate and display first appearance timestamp
                                first_appearance_time = frame_to_timestamp(frame_number, self.video_info['fps'])
                                print(f"🎯 Target person found using reference image!")
                                print(f"   📍 ID: {target_track_id}")
                                print(f"   📅 Frame: {frame_number}")
                                print(f"   ⏰ First Appearance: {first_appearance_time}")
                                
                                # Set target person for tracking
                                self.tracker.set_target_person(target_track_id, self.reference_embedding)
                                
                                # Store first appearance info for overlay
                                self.target_first_detection_frame = frame_number
                                self.target_first_detection_time = first_appearance_time
                        elif frame_number % 30 == 0:  # Progress update every 30 frames
                            print(f"Frame {frame_number}: Still searching for target person...")
                    
                    elif (self.args.manual_select or self.reference_embedding is None) and not manual_selection_done and tracked_objects:
                        # Manual selection mode - wait for multiple people or sufficient time
                        min_people_for_selection = 2  # Wait for at least 2 people
                        min_frames_before_selection = 90  # Wait at least 3 seconds at 30fps
                        
                        if len(tracked_objects) >= min_people_for_selection or frame_number >= min_frames_before_selection:
                            print(f"Frame {frame_number}: {len(tracked_objects)} persons detected. Please select target person...")
                            target_id = self.tracker.manual_select_target(display_frame, tracked_objects)
                            manual_selection_done = True
                        elif frame_number % 30 == 0:  # Progress update every 30 frames
                            print(f"Frame {frame_number}: Waiting for people to appear... ({len(tracked_objects)} person(s) detected so far)")
                        
                        if target_id is not None:
                            target_found = True
                            results['target_person']['track_id'] = target_id
                            results['target_person']['first_seen_frame'] = frame_number
                            results['target_person']['first_seen_timestamp'] = frame_to_timestamp(
                                frame_number, self.video_info['fps']
                            )
                            # Calculate and display first appearance timestamp
                            first_appearance_time = frame_to_timestamp(frame_number, self.video_info['fps'])
                            print(f"🎯 Target person selected manually!")
                            print(f"   📍 ID: {target_id}")
                            print(f"   📅 Frame: {frame_number}")
                            print(f"   ⏰ First Appearance: {first_appearance_time}")
                            
                            # Store first appearance info for overlay
                            self.target_first_detection_frame = frame_number
                            self.target_first_detection_time = first_appearance_time
                            
                            # Compute embedding for manually selected target for validation
                            target_bbox = None
                            for x1, y1, x2, y2, track_id in tracked_objects:
                                if track_id == target_id:
                                    target_bbox = (x1, y1, x2, y2)
                                    break
                            
                            if target_bbox:
                                try:
                                    crop = self.detector.extract_person_crop(display_frame, target_bbox)
                                    manual_embedding = self.detector.compute_embedding(crop)
                                    self.tracker.set_target_person(target_id, manual_embedding)
                                    print("   ✅ Embedding computed for validation")
                                except Exception as e:
                                    print(f"   ⚠️  Could not compute embedding: {e}")
                                    self.tracker.set_target_person(target_id)
                        else:
                            print("No person selected. Will try again in next frame with detected persons.")
                
                # Validate target person if we have one and a reference embedding
                if (target_found and self.tracker.target_person_id is not None and 
                    self.tracker.target_person_embedding is not None):
                    is_target_valid = self.tracker.validate_target_person(tracked_objects, display_frame, self.detector)
                    
                    if not is_target_valid:
                        # Try to recover target person
                        print(f"⚠️  Target validation failed at frame {frame_number}")
                        recovered_id = self.tracker.try_recover_target_person(tracked_objects, display_frame, self.detector)
                        
                        if recovered_id is not None:
                            print(f"✅ Target recovered with new ID: {recovered_id}")
                        else:
                            print("❌ Target could not be recovered - resetting target tracking")
                            # Reset target but don't break - continue tracking all people
                            self.tracker.reset_target()
                            target_found = False
                            # Could trigger manual selection again if needed
                
                # Get visualization
                if self.view_mode == "both":
                    # Create both views
                    box_frame = self.visualizer.draw_all_tracks(
                        display_frame, tracked_objects, 
                        target_person_id=self.tracker.target_person_id,
                        view_mode="box",
                        complete_target_trajectory=None  # No trajectory for box view
                    )
                    
                    flow_frame = self.visualizer.draw_all_tracks(
                        display_frame, tracked_objects,
                        trajectories=self.tracker.get_all_trajectories(),
                        target_person_id=self.tracker.target_person_id,
                        view_mode="flow",
                        complete_target_trajectory=self.tracker.get_complete_target_trajectory()
                    )
                    
                    # Combine side by side
                    output_frame = self.visualizer.create_side_by_side_view(
                        box_frame, flow_frame
                    )
                else:
                    # Single view mode
                    trajectories = self.tracker.get_all_trajectories() if self.view_mode in ["flow", "trail"] else None
                    complete_trajectory = self.tracker.get_complete_target_trajectory() if self.view_mode in ["flow", "trail"] else None
                    output_frame = self.visualizer.draw_all_tracks(
                        display_frame, tracked_objects,
                        trajectories=trajectories,
                        target_person_id=self.tracker.target_person_id,
                        view_mode=self.view_mode,
                        complete_target_trajectory=complete_trajectory
                    )
                
                # Add target detected notification if recently detected
                if (self.target_first_detection_frame is not None and 
                    self.target_first_detection_time is not None):
                    frames_since_detection = frame_number - self.target_first_detection_frame
                    output_frame = self.visualizer.add_target_detected_notification(
                        output_frame, self.tracker.target_person_id,
                        self.target_first_detection_time, frames_since_detection
                    )
                
                # Add info overlay
                target_info = self.tracker.get_target_info()
                output_frame = self.visualizer.add_info_overlay(
                    output_frame, frame_number, self.video_info['fps'],
                    target_info, self.video_info['frame_count']
                )
                
                # Write frame to output video
                if writer is not None:
                    # Resize frame to match output video resolution
                    output_frame_resized = cv2.resize(output_frame, (output_width, output_height))
                    
                    writer.write(output_frame_resized)
                
                # Display progress
                if frame_number % 30 == 0:  # Every 30 frames
                    progress_callback(frame_number, self.video_info['frame_count'], "Processing video")
                
                # Optional: Display frame (for debugging)
                if self.args.display:
                    cv2.imshow("Person Tracking", cv2.resize(output_frame, (960, 540)))
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
        
        finally:
            # Cleanup
            cap.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()
        
        # Final progress update
        results['processing_info']['total_frames_processed'] = frame_number
        progress_callback(frame_number, self.video_info['frame_count'], "Processing video")
        
        # Save results
        if target_found:
            # Save tracking results JSON
            results_path = output_path.replace('.mp4', '_results.json') if output_path else 'results.json'
            save_tracking_results(results_path, results)
            
            # Save trajectory CSV
            target_trajectory = []
            for x, y, frame in [(x, y, frame) for x, y, frame in 
                               [(x, y, i) for i, (x, y) in enumerate(self.tracker.get_target_trajectory(), 1)]]:
                target_trajectory.append((x, y, frame))
            
            if target_trajectory:
                csv_path = output_path.replace('.mp4', '_trajectory.csv') if output_path else 'trajectory.csv'
                save_trajectory_csv(
                    csv_path, target_trajectory, 
                    self.video_info['fps'], self.tracker.target_person_id
                )
                
                print(f"\nResults saved:")
                print(f"- Video: {output_path}")
                print(f"- Results: {results_path}")
                print(f"- Trajectory: {csv_path}")
        
        return True
    
    def run(self):
        """Run the complete tracking application."""
        print("Person Tracking Application Started")
        print("=" * 50)
        
        # Handle video input
        video_path = self.args.video
        if not video_path:
            # Use GUI to select video
            print("No video specified. Opening file dialog...")
            root = tk.Tk()
            root.withdraw()  # Hide main window
            video_path = filedialog.askopenfilename(
                title="Select Video File",
                filetypes=[
                    ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                    ("All files", "*.*")
                ]
            )
            root.destroy()
            
            if not video_path:
                print("No video selected. Exiting.")
                return False
        
        print(f"Input video: {video_path}")
        
        # Handle reference image - give user choice
        if self.args.ref_image:
            print(f"Reference image provided: {self.args.ref_image}")
            if not self.load_reference_image(self.args.ref_image):
                print("Failed to load reference image.")
                choice = input("Do you want to continue with manual selection instead? (y/n): ").lower().strip()
                if choice != 'y':
                    return False
                print("Continuing with manual selection mode...")
            else:
                print("✓ Reference image loaded successfully - will auto-detect target person")
        elif not self.args.manual_select:
            # If no reference image and no manual select flag, ask user
            print("\nNo reference image provided.")
            print("Choose tracking method:")
            print("1. Manual selection (click on person in video)")
            print("2. Provide reference image")
            choice = input("Enter choice (1 or 2): ").strip()
            
            if choice == "2":
                # Ask for reference image
                print("Opening file dialog for reference image...")
                root = tk.Tk()
                root.withdraw()
                ref_path = filedialog.askopenfilename(
                    title="Select Reference Image of Target Person",
                    filetypes=[
                        ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                        ("All files", "*.*")
                    ]
                )
                root.destroy()
                
                if ref_path and self.load_reference_image(ref_path):
                    print("✓ Reference image loaded successfully")
                else:
                    print("Failed to load reference image. Switching to manual selection.")
                    self.args.manual_select = True
            else:
                self.args.manual_select = True
                print("✓ Manual selection mode enabled")
        
        # Generate output path if not specified
        output_path = self.args.output
        if not output_path:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_filename = create_output_filename(f"{video_name}_tracked", self.view_mode)
            output_path = os.path.join(Config.OUTPUT_DIR, output_filename)
        
        print(f"Output video: {output_path}")
        print(f"View mode: {self.view_mode}")
        print(f"Manual selection: {self.args.manual_select}")
        
        # Process video
        success = self.process_video(video_path, output_path)
        
        if success:
            print("\n" + "=" * 50)
            print("🎉 Processing completed successfully!")
            print("=" * 50)
            
            # Print target person info if found
            target_info = self.tracker.get_target_info()
            if target_info['target_id'] is not None:
                print("\n🎯 TARGET PERSON SUMMARY:")
                print(f"   📍 Target ID: {target_info['target_id']}")
                if target_info['first_seen_frame']:
                    timestamp = frame_to_timestamp(target_info['first_seen_frame'], self.video_info['fps'])
                    print(f"   ⏰ First Appearance: Frame {target_info['first_seen_frame']} at {timestamp}")
                print(f"   📊 Trajectory Length: {target_info['trajectory_length']} tracking points")
                
                # Add validation stats if available
                if 'frames_tracked' in target_info and target_info['frames_tracked'] > 0:
                    print(f"   ✅ Validation: Tracked for {target_info['frames_tracked']} frames")
                    if 'avg_confidence' in target_info:
                        print(f"   📈 Average Confidence: {target_info['avg_confidence']:.3f}")
            else:
                print("\n⚠️  No target person was identified.")
        else:
            print("Processing failed!")
            return False
        
        return True


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="Person Tracking with Flow Line Visualization")
    
    # Input/Output arguments
    parser.add_argument('--video', '-v', type=str, help='Input video file path')
    parser.add_argument('--output', '-o', type=str, help='Output video file path')
    parser.add_argument('--ref_image', '-r', type=str, help='Reference image of target person')
    
    # Processing options
    parser.add_argument('--view', choices=['box', 'flow', 'both'], default='flow',
                       help='View mode: box (bounding boxes only), flow (with trajectories), both (side-by-side)')
    parser.add_argument('--confidence', type=float, default=Config.CONFIDENCE_THRESHOLD,
                       help='Detection confidence threshold')
    parser.add_argument('--manual_select', action='store_true',
                       help='Enable manual person selection')
    
    # Display options
    parser.add_argument('--display', action='store_true',
                       help='Display video while processing (for debugging)')
    parser.add_argument('--fps', type=float, help='Output video FPS (default: same as input)')
    
    args = parser.parse_args()
    
    # Create and run application
    try:
        app = PersonTrackingApp(args)
        success = app.run()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
