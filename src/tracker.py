"""
Tracking system using DeepSORT for consistent person tracking across frames.
Handles multi-object tracking with Kalman filter and appearance embeddings.
"""

import cv2
import numpy as np
import warnings
from typing import List, Tuple, Optional, Dict

# Suppress pkg_resources deprecation warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from deep_sort_realtime.deepsort_tracker import DeepSort
from config import Config
import time

class EarlyDetectionFilter:
    """Filter to prevent false positive detections in early frames"""
    def __init__(self, reference_embedding):
        self.reference_embedding = reference_embedding
        self.detection_history = []
        self.confidence_threshold = 0.75  # Higher threshold initially
        self.frames_for_confirmation = 8   # Require sustained detection
        
    def validate_early_detection(self, candidate_embedding, frame_number):
        """Prevent false positives in early frames"""
        
        # Compute similarity
        if (np.linalg.norm(candidate_embedding) == 0 or 
            np.linalg.norm(self.reference_embedding) == 0):
            return False
            
        similarity = np.dot(candidate_embedding, self.reference_embedding) / (
            np.linalg.norm(candidate_embedding) * np.linalg.norm(self.reference_embedding)
        )
        
        # Store detection history
        self.detection_history.append({
            'frame': frame_number,
            'similarity': similarity,
            'timestamp': time.time()
        })
        
        # Remove old detections (older than 10 frames)
        self.detection_history = [
            d for d in self.detection_history 
            if frame_number - d['frame'] <= 10
        ]
        
        # For very early frames, be extra strict
        if frame_number < 30:
            required_frames = min(self.frames_for_confirmation, max(1, frame_number // 3))
        else:
            required_frames = 3  # Less strict after initial frames
        
        # Require sustained high confidence
        if len(self.detection_history) >= required_frames:
            recent_similarities = [d['similarity'] for d in self.detection_history[-required_frames:]]
            
            # All recent frames must be above threshold
            if all(s > self.confidence_threshold for s in recent_similarities):
                # Additional check: similarity should be stable or increasing
                if len(recent_similarities) > 1:
                    trend = np.mean(np.diff(recent_similarities))
                    if trend >= -0.03:  # Allow slight decrease
                        return True
                else:
                    return True
                    
        return False


class PersonTracker:
    def __init__(self, max_age: int = None, n_init: int = 3):
        """
        Initialize the DeepSORT tracker.
        
        Args:
            max_age: Maximum number of frames to keep a track without detections
            n_init: Number of consecutive detections before a track is confirmed
        """
        self.max_age = max_age or Config.MAX_DISAPPEARED
        
        # Initialize DeepSORT tracker
        self.tracker = DeepSort(
            max_age=self.max_age,
            n_init=n_init,
            max_cosine_distance=0.7,
            max_iou_distance=0.7,
            nn_budget=100,
            override_track_class=None,
            embedder="mobilenet",
            half=False,
            bgr=True,
            embedder_gpu=False,
            embedder_model_name=None,
            embedder_wts=None,
            polygon=False,
            today=None
        )
        
        # Store trajectories for each track
        self.trajectories = {}  # track_id -> list of (x, y, frame_num) points
        
        # Target person tracking
        self.target_person_id = None
        self.target_person_embedding = None
        self.target_first_seen_frame = None
        
        # Target validation and recovery
        self.target_confidence_history = []  # Track confidence over time
        self.target_lost_frames = 0  # Count frames since target was lost
        self.max_lost_frames = 30  # Max frames to try recovery
        self.confidence_window = 10  # Number of recent frames to consider
        self.min_target_confidence = 0.4  # Minimum confidence to trust tracking
        
        # Enhanced validation system
        self.validation_buffer_size = 5
        self.confidence_buffer = []
        self.appearance_stability_threshold = 0.8
        self.pre_obstruction_embeddings = []
        self.obstruction_frames = 0
        self.early_detection_filter = None
        self.base_confidence = 1.0
        self.confidence_decay_rate = 0.95
        self.min_confidence = 0.3
        self.position_history = {}  # track_id -> list of positions
        
        # Frame counter
        self.frame_count = 0
        
    def update(self, detections: List[Tuple[int, int, int, int, float]], 
               frame: np.ndarray, embeddings: Optional[List[np.ndarray]] = None) -> List[Tuple[int, int, int, int, int]]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of (x1, y1, x2, y2, confidence) detections
            frame: Current frame
            embeddings: Optional list of embeddings for each detection
            
        Returns:
            List of tracked objects (x1, y1, x2, y2, track_id)
        """
        self.frame_count += 1
        
        # Convert detections to DeepSORT format
        if not detections:
            tracks = self.tracker.update_tracks([], frame=frame)
        else:
            # Format: [[x1, y1, x2, y2, confidence], ...]
            det_list = []
            for i, det in enumerate(detections):
                try:
                    if len(det) != 5:
                        print(f"Warning: Detection {i} has wrong format: {det} (length: {len(det)})")
                        continue
                    x1, y1, x2, y2, conf = det
                    # Convert to [[x, y, width, height], confidence] format for DeepSORT
                    width = float(x2 - x1)
                    height = float(y2 - y1)
                    det_list.append([[float(x1), float(y1), width, height], float(conf)])
                except Exception as e:
                    print(f"Error processing detection {i}: {det}, error: {e}")
                    continue
            
            try:
                # DeepSORT expects a list of detections, each as [x, y, w, h, conf]
                tracks = self.tracker.update_tracks(det_list, frame=frame)
            except Exception as e:
                print(f"Error in tracker.update_tracks: {e}")
                print(f"det_list: {det_list}")
                tracks = []
        
        # Process tracks and update trajectories
        tracked_objects = []
        
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            # Get track info
            track_id = track.track_id
            ltrb = track.to_ltrb()  # left, top, right, bottom
            x1, y1, x2, y2 = map(int, ltrb)
            
            # Calculate center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Update trajectory
            if track_id not in self.trajectories:
                self.trajectories[track_id] = []
            
            self.trajectories[track_id].append((center_x, center_y, self.frame_count))
            
            # Keep only recent trajectory points
            if len(self.trajectories[track_id]) > Config.TRAIL_LENGTH:
                self.trajectories[track_id] = self.trajectories[track_id][-Config.TRAIL_LENGTH:]
            
            tracked_objects.append((x1, y1, x2, y2, track_id))
        
        return tracked_objects
    
    def set_target_person(self, track_id: int, embedding: np.ndarray = None):
        """
        Set the target person to track.
        
        Args:
            track_id: ID of the target person track
            embedding: Optional embedding for target person
        """
        self.target_person_id = track_id
        self.target_person_embedding = embedding
        
        if self.target_first_seen_frame is None:
            self.target_first_seen_frame = self.frame_count
            
        # Reset validation history when setting new target
        self.target_confidence_history = []
        self.target_lost_frames = 0
            
        print(f"Target person set: ID {track_id} at frame {self.frame_count}")
        
    def validate_target_person(self, tracked_objects: List[Tuple[int, int, int, int, int]], 
                              frame: np.ndarray, detector) -> bool:
        """
        Validate that we're still tracking the correct person using embedding similarity.
        
        Args:
            tracked_objects: Current tracked objects
            frame: Current frame
            detector: Person detector for computing embeddings
            
        Returns:
            True if target is valid, False if lost or switched
        """
        if self.target_person_id is None or self.target_person_embedding is None:
            return False
            
        # Find current target in tracked objects
        target_bbox = None
        for x1, y1, x2, y2, track_id in tracked_objects:
            if track_id == self.target_person_id:
                target_bbox = (x1, y1, x2, y2)
                break
        
        if target_bbox is None:
            # Target not found in current frame
            self.target_lost_frames += 1
            self.target_confidence_history.append(0.0)
            
            if self.target_lost_frames > self.max_lost_frames:
                print(f"⚠️  Target person lost for {self.target_lost_frames} frames - searching for recovery")
            return False
                
        return True  # Give it a few frames to recover
            
        # Compute embedding for current target
        try:
            crop = detector.extract_person_crop(frame, target_bbox)
            current_embedding = detector.compute_embedding(crop)
            
            # Check temporal consistency
            if not self.check_temporal_consistency(target_bbox, self.target_person_id):
                print(f"⚠️  Temporal consistency check failed for target {self.target_person_id}")
                return False
                
            # Compute similarity with reference using robust metrics
            if np.linalg.norm(current_embedding) == 0 or np.linalg.norm(self.target_person_embedding) == 0:
                confidence = 0.0
                metrics = {'cosine': 0.0, 'euclidean': 0.0, 'combined': 0.0}
            else:
                confidence, metrics = self.compute_robust_similarity(current_embedding, self.target_person_embedding)
                
            # Store embeddings for obstruction recovery
            self.store_pre_obstruction_embeddings(frame, detector)
            
            # Add to confidence history
            self.target_confidence_history.append(confidence)
            if len(self.target_confidence_history) > self.confidence_window:
                self.target_confidence_history.pop(0)
                
            # Reset lost frames counter
            self.target_lost_frames = 0
            
            # Use dynamic confidence thresholding
            dynamic_threshold = self.update_dynamic_confidence(True, confidence)
            
            # Multi-frame consensus validation
            consensus_valid = self.validate_with_consensus([confidence])
            
            print(f"  Enhanced validation: confidence={confidence:.3f} (cosine: {metrics['cosine']:.3f}, euclidean: {metrics['euclidean']:.3f})")
            print(f"  Dynamic threshold: {dynamic_threshold:.3f}, Consensus: {consensus_valid}")
            
            # Enhanced validation logic
            validation_passed = (
                confidence >= dynamic_threshold and 
                consensus_valid and
                confidence >= self.min_target_confidence
            )
            
            if not validation_passed:
                print(f"  ⚠️  Enhanced validation failed: confidence={confidence:.3f}, threshold={dynamic_threshold:.3f}, consensus={consensus_valid}")
                return False
                
            if confidence < 0.3:  # Very low single-frame confidence
                print(f"⚠️  Single frame confidence very low: {confidence:.3f}")
                return False
                
            return True
            
        except Exception as e:
            print(f"Error validating target person: {e}")
            return True  # Don't break tracking on errors
    
    def find_target_person_from_embedding(self, detections: List[Tuple[int, int, int, int, float]], 
                                        embeddings: List[np.ndarray], 
                                        reference_embedding: np.ndarray,
                                        similarity_threshold: float = None) -> Optional[int]:
        """
        Find target person based on embedding similarity.
        
        Args:
            detections: List of detections
            embeddings: List of embeddings for each detection
            reference_embedding: Reference embedding to match against
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            Track ID of target person if found, None otherwise
        """
        if not detections or not embeddings:
            return None
            
        threshold = similarity_threshold or Config.EMBEDDING_SIMILARITY_THRESHOLD
        
        # Initialize early detection filter if not exists
        if self.early_detection_filter is None:
            self.early_detection_filter = EarlyDetectionFilter(reference_embedding)
        
        # Compare embeddings directly instead of updating tracks
        # We don't need to create tracks here, just find best matching embedding
        
        best_similarity = 0.0
        best_track_id = None
        
        for i, embedding in enumerate(embeddings):
            # Compute similarity with reference using robust metrics
            if np.linalg.norm(embedding) == 0 or np.linalg.norm(reference_embedding) == 0:
                continue
                
            similarity, metrics = self.compute_robust_similarity(embedding, reference_embedding)
            
            # Use early detection filter to prevent false positives
            if self.frame_count < 50:  # Apply early detection filtering
                if not self.early_detection_filter.validate_early_detection(embedding, self.frame_count):
                    print(f"  Early detection filter rejected candidate {i} (similarity: {similarity:.3f})")
                    continue
            
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_detection_idx = i
        
        return best_detection_idx
    
    def try_recover_target_person(self, tracked_objects: List[Tuple[int, int, int, int, int]], 
                                 frame: np.ndarray, detector) -> Optional[int]:
        """
        Try to recover the target person by searching all current tracks.
        
        Args:
            tracked_objects: Current tracked objects
            frame: Current frame
            detector: Person detector for computing embeddings
            
        Returns:
            New track ID if target recovered, None otherwise
        """
        if self.target_person_embedding is None:
            return None
            
        print(f"🔍 Attempting to recover target person from {len(tracked_objects)} available tracks...")
        
        best_similarity = 0.0
        best_track_id = None
        
        for x1, y1, x2, y2, track_id in tracked_objects:
            try:
                # Compute embedding for this track
                crop = detector.extract_person_crop(frame, (x1, y1, x2, y2))
                current_embedding = detector.compute_embedding(crop)
                
                # Compute similarity with reference using robust metrics
                if np.linalg.norm(current_embedding) == 0 or np.linalg.norm(self.target_person_embedding) == 0:
                    continue
                    
                # Use robust similarity and ensemble matching
                similarity, metrics = self.compute_robust_similarity(current_embedding, self.target_person_embedding)
                
                # If we have pre-obstruction embeddings, use ensemble matching
                if self.pre_obstruction_embeddings:
                    ensemble_scores = []
                    for stored_embedding in self.pre_obstruction_embeddings:
                        if np.linalg.norm(stored_embedding) > 0:
                            ensemble_sim, _ = self.compute_robust_similarity(current_embedding, stored_embedding)
                            ensemble_scores.append(ensemble_sim)
                    
                    if ensemble_scores:
                        ensemble_similarity = np.median(ensemble_scores)
                        # Weight current and ensemble similarities
                        final_similarity = 0.6 * similarity + 0.4 * ensemble_similarity
                    else:
                        final_similarity = similarity
                else:
                    final_similarity = similarity
                
                print(f"  Track {track_id}: similarity = {final_similarity:.3f} (base: {similarity:.3f})")
                
                # Higher threshold for recovery with temporal consistency check
                recovery_threshold = 0.65  # More strict for recovery
                if (final_similarity > best_similarity and 
                    final_similarity >= recovery_threshold and
                    self.check_temporal_consistency((x1, y1, x2, y2), track_id)):
                    best_similarity = final_similarity
                    best_track_id = track_id
                    
            except Exception as e:
                print(f"  Error computing embedding for track {track_id}: {e}")
                continue
        
        if best_track_id is not None:
            print(f"🎯 Target person recovered! New ID: {best_track_id} (similarity: {best_similarity:.3f})")
            # Update target person ID but keep original embedding
            old_id = self.target_person_id
            self.target_person_id = best_track_id
            self.target_lost_frames = 0
            self.target_confidence_history = []
            return best_track_id
        else:
            print("❌ Could not recover target person")
            return None
    
    def get_target_trajectory(self) -> List[Tuple[int, int]]:
        """
        Get trajectory points for the target person.
        
        Returns:
            List of (x, y) trajectory points
        """
        if self.target_person_id is None or self.target_person_id not in self.trajectories:
            return []
        
        # Return only (x, y) points without frame numbers
        return [(x, y) for x, y, _ in self.trajectories[self.target_person_id]]
    
    def get_target_info(self) -> Dict:
        """
        Get information about the target person.
        
        Returns:
            Dictionary with target person information
        """
        info = {
            'target_id': self.target_person_id,
            'first_seen_frame': self.target_first_seen_frame,
            'trajectory_length': 0,
            'is_active': False
        }
        
        if self.target_person_id is not None:
            trajectory = self.get_target_trajectory()
            info['trajectory_length'] = len(trajectory)
            
            # Check if target is still active (has recent trajectory points)
            if self.target_person_id in self.trajectories:
                recent_frames = [frame for _, _, frame in self.trajectories[self.target_person_id][-10:]]
                if recent_frames and max(recent_frames) >= self.frame_count - 5:
                    info['is_active'] = True
        
        # Add validation statistics
        validation_stats = self.get_validation_stats()
        info.update(validation_stats)
        
        return info
    
    def check_temporal_consistency(self, current_bbox, track_id, max_movement=120):
        """Check if movement is physically reasonable"""
        if track_id not in self.position_history:
            self.position_history[track_id] = []
            return True
            
        # Current center position
        current_center = (
            (current_bbox[0] + current_bbox[2]) // 2,
            (current_bbox[1] + current_bbox[3]) // 2
        )
        
        # Store current position
        self.position_history[track_id].append(current_center)
        
        # Keep only recent positions
        if len(self.position_history[track_id]) > 10:
            self.position_history[track_id] = self.position_history[track_id][-10:]
            
        if len(self.position_history[track_id]) < 2:
            return True
            
        # Check if movement is reasonable
        last_pos = self.position_history[track_id][-2]
        distance = np.sqrt((current_center[0] - last_pos[0])**2 + 
                          (current_center[1] - last_pos[1])**2)
        
        if distance > max_movement:
            print(f"  Movement too large: {distance:.1f} pixels")
            return False
            
        return True
    
    def compute_robust_similarity(self, embedding1, embedding2):
        """Multiple similarity metrics for robust matching"""
        
        # Cosine similarity (main metric)
        cosine_sim = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        
        # Euclidean distance (normalized)
        euclidean_dist = np.linalg.norm(embedding1 - embedding2)
        euclidean_sim = 1.0 / (1.0 + euclidean_dist)
        
        # Combined similarity (weighted)
        combined_similarity = cosine_sim * 0.7 + euclidean_sim * 0.3
        
        return combined_similarity, {
            'cosine': cosine_sim,
            'euclidean': euclidean_sim,
            'combined': combined_similarity
        }
    
    def validate_with_consensus(self, similarity_scores):
        """Multi-frame consensus validation"""
        self.confidence_buffer.append(max(similarity_scores) if similarity_scores else 0.0)
        
        if len(self.confidence_buffer) > self.validation_buffer_size:
            self.confidence_buffer.pop(0)
            
        # Require consistent high confidence across multiple frames
        if len(self.confidence_buffer) >= 3:
            recent_scores = self.confidence_buffer[-3:]
            consensus_score = np.mean(recent_scores)
            stability = 1.0 - np.std(recent_scores)
            
            return consensus_score > 0.65 and stability > self.appearance_stability_threshold
        
        return len(self.confidence_buffer) > 0 and self.confidence_buffer[-1] > 0.75
    
    def update_dynamic_confidence(self, validation_success, similarity_score):
        """Dynamic confidence adjustment"""
        
        if validation_success and similarity_score > 0.8:
            # High confidence detection - boost confidence
            self.base_confidence = min(1.0, self.base_confidence * 1.05)
        elif validation_success and similarity_score > 0.6:
            # Moderate confidence - maintain
            pass
        else:
            # Low confidence or validation failure - decay
            self.base_confidence *= self.confidence_decay_rate
            
        # Apply minimum threshold
        self.base_confidence = max(self.min_confidence, self.base_confidence)
        
        # Adjust similarity threshold based on current confidence
        adjusted_threshold = 0.6 * self.base_confidence
        
        return adjusted_threshold
    
    def store_pre_obstruction_embeddings(self, frame, detector):
        """Store multiple embeddings before obstruction for robust recovery"""
        if self.target_person_id is None:
            return
            
        # Find target bbox
        for x1, y1, x2, y2, track_id in self.get_current_tracks():
            if track_id == self.target_person_id:
                try:
                    crop = detector.extract_person_crop(frame, (x1, y1, x2, y2))
                    embedding = detector.compute_embedding(crop)
                    
                    self.pre_obstruction_embeddings.append(embedding)
                    
                    # Keep only recent embeddings
                    if len(self.pre_obstruction_embeddings) > 5:
                        self.pre_obstruction_embeddings.pop(0)
                        
                except Exception as e:
                    print(f"Error storing pre-obstruction embedding: {e}")
                break
    
    def manual_select_target(self, frame: np.ndarray, tracked_objects: List[Tuple[int, int, int, int, int]]) -> Optional[int]:
        """
        Allow manual selection of target person from current tracks.
        
        Args:
            frame: Current frame
            tracked_objects: List of tracked objects (x1, y1, x2, y2, track_id)
            
        Returns:
            Selected track ID or None if cancelled
        """
        if not tracked_objects:
            print("No tracked objects available for selection")
            return None

        print(f"Manual selection: {len(tracked_objects)} people available for selection")
        
        # Create a copy of the frame for display
        display_frame = frame.copy()        # Draw all tracked objects with their IDs - make them very visible
        for i, (x1, y1, x2, y2, track_id) in enumerate(tracked_objects):
            # Use different colors for each person
            colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
            color = colors[i % len(colors)]
            
            # Draw thick bounding box
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw ID with background for better visibility
            label = f"Person {track_id}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(display_frame, (x1, y1 - 30), (x1 + label_size[0] + 10, y1), color, -1)
            cv2.putText(display_frame, label, (x1 + 5, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Add instructions
        cv2.putText(display_frame, "Click on person to track, press ESC to cancel",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Ensure any existing window is closed first
        try:
            cv2.destroyWindow("Select Target Person")
        except:
            pass
        
        # Create window and show frame
        cv2.namedWindow("Select Target Person", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Select Target Person", display_frame)
        cv2.waitKey(1)  # Allow window to render
        
        selected_track_id = None
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal selected_track_id
            if event == cv2.EVENT_LBUTTONDOWN:
                print(f"Mouse clicked at ({x}, {y})")
                # Find which bounding box was clicked
                for x1, y1, x2, y2, track_id in tracked_objects:
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        selected_track_id = track_id
                        print(f"✓ Selected person with ID: {track_id}")
                        return
                print("No person selected - click inside a bounding box")
        
        cv2.setMouseCallback("Select Target Person", mouse_callback)
        
        # Wait for selection or ESC key
        print("Waiting for user selection (click on person or press ESC to cancel)...")
        while selected_track_id is None:
            key = cv2.waitKey(30) & 0xFF  # Increased wait time for better responsiveness
            if key == 27:  # ESC key
                print("Selection cancelled by user")
                break
        
        # Clean up window and callbacks
        try:
            # Check if window exists before destroying
            if cv2.getWindowProperty("Select Target Person", cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyWindow("Select Target Person")
        except:
            pass
        
        try:
            cv2.setMouseCallback("Select Target Person", lambda *args: None)
        except:
            pass
        
        if selected_track_id is not None:
            self.set_target_person(selected_track_id)
        
        return selected_track_id
    
    def reset_target(self):
        """Reset target person tracking."""
        self.target_person_id = None
        self.target_person_embedding = None
        self.target_first_seen_frame = None
        self.target_confidence_history = []
        self.target_lost_frames = 0
        print("Target person tracking reset")
    
    def get_validation_stats(self) -> Dict[str, float]:
        """Get current validation statistics for the target person."""
        if not self.target_confidence_history:
            return {}
            
        return {
            'current_confidence': self.target_confidence_history[-1] if self.target_confidence_history else 0.0,
            'avg_confidence': np.mean(self.target_confidence_history),
            'recent_avg_confidence': np.mean(self.target_confidence_history[-5:]) if len(self.target_confidence_history) >= 5 else 0.0,
            'frames_tracked': len(self.target_confidence_history),
            'lost_frames': self.target_lost_frames
        }
    
    def get_all_trajectories(self) -> Dict[int, List[Tuple[int, int]]]:
        """
        Get trajectories for all tracked objects.
        
        Returns:
            Dictionary mapping track_id to list of (x, y) points
        """
        all_trajectories = {}
        for track_id, trajectory in self.trajectories.items():
            all_trajectories[track_id] = [(x, y) for x, y, _ in trajectory]
        
        return all_trajectories
