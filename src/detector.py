"""
Person detection module using YOLOv8.
Handles detection of persons in video frames and extraction of embeddings for tracking.
"""

import cv2
import numpy as np
import warnings
from ultralytics import YOLO
from typing import List, Tuple, Optional, Dict
import torch
import os
import logging

# Suppress deprecation warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from config import Config


class PersonDetector:
    def __init__(self, model_path: str = None, confidence_threshold: float = None):
        """
        Initialize the person detector with YOLOv8.
        
        Args:
            model_path: Path to YOLOv8 model file
            confidence_threshold: Confidence threshold for detections
        """
        self.model_path = model_path or Config.YOLO_MODEL
        self.confidence_threshold = confidence_threshold or Config.CONFIDENCE_THRESHOLD
        
        # Load YOLOv8 model
        try:
            self.model = YOLO(self.model_path)
            print(f"Loaded YOLOv8 model: {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # COCO class ID for person is 0
        self.person_class_id = 0
        
        # Initialize face detector using OpenCV's pre-trained cascade
        try:
            # Try to load face cascade from OpenCV data
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                raise Exception("Could not load face cascade")
            print("Face detection initialized successfully")
        except Exception as e:
            print(f"Warning: Face detection not available: {e}")
            self.face_cascade = None
        
    def detect_persons(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect persons in a frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            List of tuples containing (x1, y1, x2, y2, confidence) for each person
        """
        # Run YOLOv8 detection
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get class ID and confidence
                    class_id = int(box.cls.cpu().numpy()[0])
                    confidence = float(box.conf.cpu().numpy()[0])
                    
                    # Only keep person detections
                    if class_id == self.person_class_id:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].astype(int)
                        detections.append((int(x1), int(y1), int(x2), int(y2), float(confidence)))
        
        return detections
    
    def extract_person_crop(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract person crop from frame using bounding box.
        
        Args:
            frame: Input video frame
            bbox: Bounding box (x1, y1, x2, y2)
            
        Returns:
            Cropped person image
        """
        x1, y1, x2, y2 = bbox
        
        # Ensure coordinates are within frame bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        # Extract crop
        crop = frame[y1:y2, x1:x2]
        
        # Resize to standard size for consistency
        if crop.size > 0:
            crop = cv2.resize(crop, Config.REFERENCE_IMAGE_SIZE)
        
        return crop
    
    def compute_embedding(self, crop: np.ndarray) -> np.ndarray:
        """
        Compute embedding for a person crop using facial + outfit feature extraction.
        
        Extracts features from:
        1. Face region (if detected): facial structure, skin tone
        2. Upper body (torso): clothing colors and patterns  
        3. Lower body (legs): pants/skirt colors and textures
        4. Overall appearance: posture, body proportions
        
        Args:
            crop: Person crop image
            
        Returns:
            Feature embedding vector (512D)
        """
        if crop.size == 0:
            return np.zeros(512)  # Return zero embedding for empty crop
        
        features = []
        
        # === 1. FACIAL FEATURES EXTRACTION ===
        face_features = self._extract_face_features(crop)
        features.extend(face_features)  # 128 features
        
        # === 2. OUTFIT FEATURES EXTRACTION ===
        outfit_features = self._extract_outfit_features(crop)
        features.extend(outfit_features)  # 256 features
        
        # === 3. BODY STRUCTURE FEATURES ===
        body_features = self._extract_body_features(crop)
        features.extend(body_features)  # 64 features
        
        # === 4. GLOBAL APPEARANCE FEATURES ===
        global_features = self._extract_global_features(crop)
        features.extend(global_features)  # 64 features
        
        # Normalize and ensure exactly 512 features
        try:
            features_array = np.array(features, dtype=np.float32)
            
            # Ensure we have exactly 512 features
            if len(features_array) < 512:
                # Pad with zeros
                features_array = np.pad(features_array, (0, 512 - len(features_array)), 'constant')
            elif len(features_array) > 512:
                # Truncate
                features_array = features_array[:512]
            
            # L2 normalize
            norm = np.linalg.norm(features_array)
            if norm > 0:
                features_array = features_array / norm
                
            return features_array
            
        except Exception as e:
            logging.warning(f"Error in embedding computation: {e}")
            return np.zeros(512)
    
    def _extract_face_features(self, crop: np.ndarray) -> List[float]:
        """Extract facial features from the upper portion of the person crop."""
        features = []
        
        # Focus on upper 40% of the image where face is likely to be
        height = crop.shape[0]
        face_region = crop[:int(height * 0.4), :]
        
        if self.face_cascade is not None:
            try:
                # Convert to grayscale for face detection
                gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(
                    gray_face, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20)
                )
                
                if len(faces) > 0:
                    # Use the largest detected face
                    largest_face = max(faces, key=lambda f: f[2] * f[3])
                    x, y, w, h = largest_face
                    
                    # Extract face crop
                    face_crop = face_region[y:y+h, x:x+w]
                    
                    # Facial color features (skin tone)
                    face_hsv = cv2.cvtColor(face_crop, cv2.COLOR_BGR2HSV)
                    face_lab = cv2.cvtColor(face_crop, cv2.COLOR_BGR2LAB)
                    
                    # Skin tone histogram (smaller bins for precision)
                    skin_h = cv2.calcHist([face_hsv], [0], None, [8], [0, 180])
                    skin_s = cv2.calcHist([face_hsv], [1], None, [8], [0, 256])
                    skin_l = cv2.calcHist([face_lab], [0], None, [8], [0, 256])
                    
                    features.extend(skin_h.flatten())  # 8 features
                    features.extend(skin_s.flatten())  # 8 features
                    features.extend(skin_l.flatten())  # 8 features
                    
                    # Face structure features
                    face_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                    
                    # Edge density (facial structure)
                    edges = cv2.Canny(face_gray, 50, 150)
                    edge_density = np.sum(edges > 0) / edges.size
                    features.append(edge_density)
                    
                    # Face region statistics
                    features.extend([
                        np.mean(face_gray), np.std(face_gray),
                        np.mean(face_crop[:,:,0]), np.mean(face_crop[:,:,1]), np.mean(face_crop[:,:,2])
                    ])
                    
                else:
                    # No face detected, use general head region features
                    head_hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
                    head_hist = cv2.calcHist([head_hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
                    features.extend(head_hist.flatten()[:29])  # 29 features to match face case
                    
            except Exception as e:
                logging.warning(f"Error in face feature extraction: {e}")
                features = [0.0] * 29
        else:
            # Face detection not available, use color features from head region
            try:
                head_hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
                head_hist = cv2.calcHist([head_hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
                features.extend(head_hist.flatten()[:29])
            except:
                features = [0.0] * 29
        
        # Pad to exactly 128 features
        while len(features) < 128:
            features.append(0.0)
        return features[:128]
    
    def _extract_outfit_features(self, crop: np.ndarray) -> List[float]:
        """Extract clothing features from upper and lower body regions."""
        features = []
        height = crop.shape[0]
        
        try:
            # Split into upper body (torso) and lower body (legs) regions
            upper_region = crop[int(height * 0.25):int(height * 0.7), :]  # Torso area
            lower_region = crop[int(height * 0.6):, :]  # Legs area
            
            # === UPPER BODY (Shirt/Jacket) FEATURES ===
            if upper_region.size > 0:
                upper_hsv = cv2.cvtColor(upper_region, cv2.COLOR_BGR2HSV)
                upper_lab = cv2.cvtColor(upper_region, cv2.COLOR_BGR2LAB)
                
                # Detailed color histograms for clothing
                upper_h = cv2.calcHist([upper_hsv], [0], None, [24], [0, 180])
                upper_s = cv2.calcHist([upper_hsv], [1], None, [16], [0, 256])
                upper_v = cv2.calcHist([upper_hsv], [2], None, [16], [0, 256])
                upper_l = cv2.calcHist([upper_lab], [0], None, [16], [0, 256])
                upper_a = cv2.calcHist([upper_lab], [1], None, [16], [0, 256])
                upper_b = cv2.calcHist([upper_lab], [2], None, [16], [0, 256])
                
                features.extend(upper_h.flatten())  # 24 features - detailed hue
                features.extend(upper_s.flatten())  # 16 features
                features.extend(upper_v.flatten())  # 16 features
                features.extend(upper_l.flatten())  # 16 features
                features.extend(upper_a.flatten())  # 16 features
                features.extend(upper_b.flatten())  # 16 features
                
                # Texture analysis for upper body
                upper_gray = cv2.cvtColor(upper_region, cv2.COLOR_BGR2GRAY)
                upper_edges = cv2.Canny(upper_gray, 50, 150)
                upper_texture = np.sum(upper_edges > 0) / upper_edges.size
                features.append(upper_texture)
            else:
                features.extend([0.0] * 105)  # 104 + 1 texture
            
            # === LOWER BODY (Pants/Skirt) FEATURES ===
            if lower_region.size > 0:
                lower_hsv = cv2.cvtColor(lower_region, cv2.COLOR_BGR2HSV)
                lower_lab = cv2.cvtColor(lower_region, cv2.COLOR_BGR2LAB)
                
                # Color histograms for lower clothing
                lower_h = cv2.calcHist([lower_hsv], [0], None, [16], [0, 180])
                lower_s = cv2.calcHist([lower_hsv], [1], None, [12], [0, 256])
                lower_v = cv2.calcHist([lower_hsv], [2], None, [12], [0, 256])
                lower_l = cv2.calcHist([lower_lab], [0], None, [12], [0, 256])
                
                features.extend(lower_h.flatten())  # 16 features
                features.extend(lower_s.flatten())  # 12 features
                features.extend(lower_v.flatten())  # 12 features
                features.extend(lower_l.flatten())  # 12 features
                
                # Lower body texture
                lower_gray = cv2.cvtColor(lower_region, cv2.COLOR_BGR2GRAY)
                lower_edges = cv2.Canny(lower_gray, 50, 150)
                lower_texture = np.sum(lower_edges > 0) / lower_edges.size
                features.append(lower_texture)
            else:
                features.extend([0.0] * 53)  # 52 + 1 texture
                
        except Exception as e:
            logging.warning(f"Error in outfit feature extraction: {e}")
            # Return zero features if error occurs
            features = [0.0] * 158
        
        # Pad to exactly 256 features
        while len(features) < 256:
            features.append(0.0)
        return features[:256]
    
    def _extract_body_features(self, crop: np.ndarray) -> List[float]:
        """Extract body structure and proportions."""
        features = []
        
        try:
            height, width = crop.shape[:2]
            
            # Body proportions
            aspect_ratio = width / height if height > 0 else 0
            features.append(aspect_ratio)
            
            # Overall color distribution
            hsv_full = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            
            # Dominant colors (simplified)
            hist_full_h = cv2.calcHist([hsv_full], [0], None, [12], [0, 180])
            hist_full_s = cv2.calcHist([hsv_full], [1], None, [8], [0, 256])
            hist_full_v = cv2.calcHist([hsv_full], [2], None, [8], [0, 256])
            
            features.extend(hist_full_h.flatten())  # 12 features
            features.extend(hist_full_s.flatten())  # 8 features  
            features.extend(hist_full_v.flatten())  # 8 features
            
            # Overall brightness and contrast
            gray_full = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            features.extend([
                np.mean(gray_full), np.std(gray_full),
                np.min(gray_full), np.max(gray_full)
            ])
            
        except Exception as e:
            logging.warning(f"Error in body feature extraction: {e}")
            features = [0.0] * 33
        
        # Pad to exactly 64 features
        while len(features) < 64:
            features.append(0.0)
        return features[:64]
    
    def _extract_global_features(self, crop: np.ndarray) -> List[float]:
        """Extract global appearance features."""
        features = []
        
        try:
            # Convert to different color spaces
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            
            # Global statistics
            features.extend([
                np.mean(hsv[:,:,0]), np.std(hsv[:,:,0]),  # Hue stats
                np.mean(hsv[:,:,1]), np.std(hsv[:,:,1]),  # Saturation stats  
                np.mean(hsv[:,:,2]), np.std(hsv[:,:,2]),  # Value stats
                np.mean(lab[:,:,0]), np.std(lab[:,:,0]),  # Lightness stats
                np.mean(lab[:,:,1]), np.std(lab[:,:,1]),  # A channel stats
                np.mean(lab[:,:,2]), np.std(lab[:,:,2]),  # B channel stats
            ])
            
            # Texture analysis
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            features.extend([
                np.mean(grad_x), np.std(grad_x),
                np.mean(grad_y), np.std(grad_y),
            ])
            
        except Exception as e:
            logging.warning(f"Error in global feature extraction: {e}")
            features = [0.0] * 16
        
        # Pad to exactly 64 features
        while len(features) < 64:
            features.append(0.0)
        return features[:64]
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score (0-1)
        """
        if np.linalg.norm(embedding1) == 0 or np.linalg.norm(embedding2) == 0:
            return 0.0
        
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        
        # Ensure similarity is between 0 and 1
        return max(0.0, min(1.0, similarity))
    
    def load_reference_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load and process reference image.
        
        Args:
            image_path: Path to reference image
            
        Returns:
            Processed reference image or None if failed
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load reference image: {image_path}")
                return None
            
            # Store reference image for color validation
            self.reference_image = image.copy()
            
            # Resize to standard size
            image = cv2.resize(image, Config.REFERENCE_IMAGE_SIZE)
            return image
        
        except Exception as e:
            print(f"Error loading reference image: {e}")
            return None
    
    def validate_target_colors(self, crop: np.ndarray, reference_crop: np.ndarray) -> float:
        """
        Validate target person based on color characteristics from reference image.
        
        Args:
            crop: Current person crop
            reference_crop: Reference person crop
            
        Returns:
            Color validation score (0-1, higher is better match)
        """
        try:
            if crop is None or reference_crop is None:
                return 0.0
            
            # Resize both crops to same size for comparison
            target_size = (128, 256)  # width, height
            crop_resized = cv2.resize(crop, target_size)
            ref_resized = cv2.resize(reference_crop, target_size)
            
            # Convert to HSV for better color analysis
            crop_hsv = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)
            ref_hsv = cv2.cvtColor(ref_resized, cv2.COLOR_BGR2HSV)
            
            # Focus on clothing regions (lower 2/3 of image)
            clothing_start = target_size[1] // 3
            crop_clothing = crop_hsv[clothing_start:, :]
            ref_clothing = ref_hsv[clothing_start:, :]
            
            # Calculate color histograms for clothing region
            crop_hist = cv2.calcHist([crop_clothing], [0, 1], None, [32, 32], [0, 180, 0, 256])
            ref_hist = cv2.calcHist([ref_clothing], [0, 1], None, [32, 32], [0, 180, 0, 256])
            
            # Normalize histograms
            cv2.normalize(crop_hist, crop_hist, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(ref_hist, ref_hist, 0, 1, cv2.NORM_MINMAX)
            
            # Compare histograms using correlation
            color_similarity = cv2.compareHist(crop_hist, ref_hist, cv2.HISTCMP_CORREL)
            
            # Look for specific colors in reference (green jacket, blue trolley)
            ref_colors = self._extract_dominant_colors(ref_clothing)
            crop_colors = self._extract_dominant_colors(crop_clothing)
            
            # Check for green (jacket) and blue (trolley) presence
            green_score = self._check_color_presence(crop_colors, ref_colors, 'green')
            blue_score = self._check_color_presence(crop_colors, ref_colors, 'blue')
            
            # Combine scores - heavily weight the specific colors
            final_score = (color_similarity * 0.4 + green_score * 0.4 + blue_score * 0.2)
            return max(0.0, min(1.0, final_score))
            
        except Exception as e:
            logging.warning(f"Error in color validation: {e}")
            return 0.0
    
    def _extract_dominant_colors(self, hsv_region: np.ndarray) -> Dict[str, float]:
        """Extract dominant color scores from HSV region"""
        try:
            # Define color ranges in HSV
            color_ranges = {
                'green': [(40, 40, 40), (80, 255, 255)],    # Green range for jacket
                'blue': [(100, 40, 40), (130, 255, 255)],   # Blue range for trolley
                'red': [(0, 40, 40), (10, 255, 255)],       # Red range
                'yellow': [(20, 40, 40), (30, 255, 255)],   # Yellow range
            }
            
            scores = {}
            total_pixels = hsv_region.shape[0] * hsv_region.shape[1]
            
            for color_name, (lower, upper) in color_ranges.items():
                lower = np.array(lower)
                upper = np.array(upper)
                mask = cv2.inRange(hsv_region, lower, upper)
                color_pixels = np.sum(mask > 0)
                scores[color_name] = color_pixels / total_pixels if total_pixels > 0 else 0.0
                
            return scores
            
        except Exception:
            return {'green': 0.0, 'blue': 0.0, 'red': 0.0, 'yellow': 0.0}
    
    def _check_color_presence(self, crop_colors: Dict[str, float], ref_colors: Dict[str, float], 
                            color: str) -> float:
        """Check if a specific color is present in both crops"""
        crop_presence = crop_colors.get(color, 0.0)
        ref_presence = ref_colors.get(color, 0.0)
        
        # Both should have significant presence of the color
        if ref_presence > 0.05:  # Reference has this color (5% of pixels)
            if crop_presence > 0.03:  # Candidate also has this color (3% of pixels)
                # Return similarity score
                return min(1.0, (crop_presence + ref_presence) / 2.0)
        
        return 0.0
    
    def find_best_match(self, frame: np.ndarray, detections: List[Tuple[int, int, int, int, float]], 
                       reference_embedding: np.ndarray) -> Optional[Tuple[int, int, int, int, float, float]]:
        """
        Find the best matching person detection based on reference embedding.
        
        Args:
            frame: Input video frame
            detections: List of person detections
            reference_embedding: Reference person embedding
            
        Returns:
            Best matching detection with similarity score or None
        """
        best_match = None
        best_similarity = 0.0
        
        for detection in detections:
            x1, y1, x2, y2, confidence = detection
            
            # Extract person crop
            crop = self.extract_person_crop(frame, (x1, y1, x2, y2))
            
            # Compute embedding
            embedding = self.compute_embedding(crop)
            
            # Compute similarity
            similarity = self.compute_similarity(embedding, reference_embedding)
            
            # Check if this is the best match so far
            if similarity > best_similarity and similarity >= Config.EMBEDDING_SIMILARITY_THRESHOLD:
                best_similarity = similarity
                best_match = (x1, y1, x2, y2, confidence, similarity)
        
        return best_match
