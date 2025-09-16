#!/usr/bin/env python3
"""
YOLOv8 Model Comparison for Person Tracking

This script demonstrates the differences between YOLOv8 models
and allows testing with different models for optimal accuracy.
"""

import time
import cv2
from ultralytics import YOLO
from src.config import Config

class ModelComparator:
    """Compare different YOLOv8 models for person tracking accuracy"""
    
    def __init__(self):
        self.models = {
            'yolov8n.pt': {
                'name': 'YOLOv8 Nano',
                'size': '6.2MB',
                'speed': 'Very Fast',
                'accuracy': 'Basic',
                'description': 'Fastest, lowest accuracy - good for real-time on weak hardware'
            },
            'yolov8s.pt': {
                'name': 'YOLOv8 Small', 
                'size': '21.5MB',
                'speed': 'Fast',
                'accuracy': 'Good',
                'description': 'Good balance of speed and accuracy - recommended for most use cases'
            },
            'yolov8m.pt': {
                'name': 'YOLOv8 Medium',
                'size': '49.7MB', 
                'speed': 'Medium',
                'accuracy': 'High',
                'description': 'Better accuracy, slower - ideal for accuracy-critical applications'
            },
            'yolov8l.pt': {
                'name': 'YOLOv8 Large',
                'size': '83.7MB',
                'speed': 'Slow',
                'accuracy': 'Very High', 
                'description': 'High accuracy, slower processing - for best detection quality'
            },
            'yolov8x.pt': {
                'name': 'YOLOv8 Extra Large',
                'size': '136MB',
                'speed': 'Very Slow',
                'accuracy': 'Maximum',
                'description': 'Maximum accuracy, slowest - for offline processing'
            }
        }
    
    def print_comparison(self):
        """Print detailed model comparison"""
        print("🔍 YOLOv8 Model Comparison for Person Tracking")
        print("=" * 80)
        print(f"{'Model':<20} {'Size':<10} {'Speed':<12} {'Accuracy':<12} {'Best For'}")
        print("-" * 80)
        
        for model_file, info in self.models.items():
            best_for = "Speed" if "Fast" in info['speed'] else "Accuracy" if "High" in info['accuracy'] else "Balance"
            print(f"{info['name']:<20} {info['size']:<10} {info['speed']:<12} {info['accuracy']:<12} {best_for}")
        
        print("\n📊 **RECOMMENDATIONS FOR PERSON TRACKING:**")
        print("  🏃 Real-time performance: yolov8n.pt or yolov8s.pt")
        print("  ⚖️  Balanced performance: yolov8s.pt or yolov8m.pt") 
        print("  🎯 Maximum accuracy: yolov8m.pt or yolov8l.pt")
        print("  🔬 Research/Analysis: yolov8l.pt or yolov8x.pt")
        
        print(f"\n🎯 **CURRENT MODEL:** {Config.YOLO_MODEL}")
        current_info = self.models.get(Config.YOLO_MODEL, {})
        if current_info:
            print(f"   📝 {current_info['description']}")
    
    def test_model_speed(self, model_path, test_image_path=None):
        """Test inference speed of a specific model"""
        try:
            print(f"\n🔬 Testing {model_path}...")
            model = YOLO(model_path)
            
            # Create a test image if none provided
            if test_image_path is None:
                import numpy as np
                test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            else:
                test_image = cv2.imread(test_image_path)
            
            # Warmup
            for _ in range(3):
                _ = model(test_image, verbose=False)
            
            # Time inference
            times = []
            for _ in range(10):
                start = time.time()
                results = model(test_image, verbose=False)
                end = time.time()
                times.append(end - start)
            
            avg_time = sum(times) / len(times)
            fps = 1.0 / avg_time
            
            # Count person detections
            person_count = 0
            if results and len(results) > 0:
                for result in results:
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        classes = result.boxes.cls.cpu().numpy()
                        person_count = sum(1 for cls in classes if int(cls) == 0)  # 0 = person class
            
            print(f"   ⏱️  Average inference time: {avg_time:.3f}s")
            print(f"   🎬 Estimated FPS: {fps:.1f}")
            print(f"   👥 Person detections: {person_count}")
            
            return avg_time, fps, person_count
            
        except Exception as e:
            print(f"   ❌ Error testing {model_path}: {e}")
            return None, None, None
    
    def recommend_model(self, priority='balanced'):
        """Recommend best model based on priority"""
        recommendations = {
            'speed': 'yolov8n.pt',
            'balanced': 'yolov8s.pt', 
            'accuracy': 'yolov8m.pt',
            'max_accuracy': 'yolov8l.pt'
        }
        
        recommended = recommendations.get(priority, 'yolov8s.pt')
        info = self.models[recommended]
        
        print(f"\n🎯 **RECOMMENDED MODEL for {priority.upper()}:** {recommended}")
        print(f"   📝 {info['description']}")
        print(f"   📊 Size: {info['size']}, Speed: {info['speed']}, Accuracy: {info['accuracy']}")
        
        return recommended

def update_model(new_model):
    """Update the configuration to use a new model"""
    config_path = "src/config.py"
    
    try:
        # Read current config
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Replace model line
        import re
        pattern = r'YOLO_MODEL = "[^"]*"'
        replacement = f'YOLO_MODEL = "{new_model}"'
        new_content = re.sub(pattern, replacement, content)
        
        # Write back
        with open(config_path, 'w') as f:
            f.write(new_content)
        
        print(f"✅ Updated configuration to use {new_model}")
        return True
        
    except Exception as e:
        print(f"❌ Error updating config: {e}")
        return False

def main():
    """Main function to run model comparison"""
    comparator = ModelComparator()
    
    print("🚀 YOLOv8 Model Comparison Tool")
    print("=" * 50)
    
    # Show comparison
    comparator.print_comparison()
    
    print("\n" + "=" * 50)
    print("💡 **WHY UPGRADING FROM NANO HELPS:**")
    print("   🎯 Better person detection accuracy")
    print("   🔍 More precise bounding boxes") 
    print("   👥 Better handling of overlapping people")
    print("   🎨 Improved feature extraction for tracking")
    print("   ⚡ More stable embeddings for similarity matching")
    
    print("\n" + "=" * 50)
    print("🛠️  **UPGRADE RECOMMENDATIONS:**")
    print("   1. For your person tracking: yolov8m.pt (Medium) - BEST BALANCE")
    print("   2. If speed is critical: yolov8s.pt (Small)")
    print("   3. For maximum accuracy: yolov8l.pt (Large)")
    
    print(f"\n🔄 **CURRENT STATUS:** Using {Config.YOLO_MODEL}")
    if Config.YOLO_MODEL == "yolov8n.pt":
        print("   ⚠️  You're using the lowest accuracy model - consider upgrading!")
    elif Config.YOLO_MODEL == "yolov8m.pt":
        print("   ✅ Great choice! Good balance of accuracy and speed.")
    
    # Interactive model selection
    print("\n" + "=" * 50)
    choice = input("🔧 Would you like to test a different model? (y/n): ").lower().strip()
    
    if choice == 'y':
        print("\nAvailable models:")
        for i, (model_file, info) in enumerate(comparator.models.items(), 1):
            print(f"   {i}. {info['name']} ({model_file}) - {info['accuracy']} accuracy")
        
        try:
            selection = int(input("Enter model number (1-5): "))
            model_files = list(comparator.models.keys())
            if 1 <= selection <= len(model_files):
                new_model = model_files[selection - 1]
                if update_model(new_model):
                    print(f"\n🎉 Configuration updated! Restart your tracking system to use {new_model}")
                    print("   The new model will be automatically downloaded on first use.")
            else:
                print("❌ Invalid selection")
        except ValueError:
            print("❌ Invalid input")

if __name__ == "__main__":
    main()