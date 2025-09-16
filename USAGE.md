# Person Tracker - Usage Guide

## Quick Start

### 1. Setup (First Time Only)

**Windows:**
```cmd
run.bat setup
```

**Linux/Mac:**
```bash
./run.sh setup
```

### 2. Basic Usage - Manual Selection

**Windows:**
```cmd
run.bat run
```

**Linux/Mac:**
```bash
./run.sh run
```

This will:
1. Open a file dialog to select your video
2. Show the first frame with detected persons
3. Allow you to click on the person you want to track
4. Generate a tracked video with flow lines

### 3. Using Reference Image

**Windows:**
```cmd
run.bat run-ref videos\your_video.mp4 images\reference_person.jpg
```

**Linux/Mac:**
```bash
./run.sh run-ref videos/your_video.mp4 images/reference_person.jpg
```

### 4. Web Interface

**Windows:**
```cmd
run.bat web
```

**Linux/Mac:**
```bash
./run.sh web
```

Then open your browser to `http://localhost:5000`

## Command Line Options

```bash
python src/main.py [OPTIONS]

Options:
  --video, -v          Input video file path
  --output, -o         Output video file path  
  --ref_image, -r      Reference image of target person
  --view               View mode: 'box', 'flow', or 'both'
  --confidence         Detection confidence (0.1-1.0)
  --manual_select      Enable manual person selection
  --display            Show video while processing
  --fps               Output video FPS
```

## Examples

### Example 1: Basic tracking with manual selection
```bash
python src/main.py --video input.mp4 --manual_select --view flow
```

### Example 2: Tracking with reference image
```bash
python src/main.py --video input.mp4 --ref_image person.jpg --view flow
```

### Example 3: Side-by-side comparison
```bash
python src/main.py --video input.mp4 --view both --manual_select
```

### Example 4: Bounding box only
```bash
python src/main.py --video input.mp4 --view box --confidence 0.7
```

## View Modes

- **`flow`**: Shows bounding boxes + trajectory lines (default)
- **`box`**: Shows only bounding boxes
- **`both`**: Side-by-side comparison of box and flow views

## Output Files

After processing, you'll get:

1. **Tracked Video** (`.mp4`): Video with visualizations
2. **Results JSON** (`_results.json`): Metadata and timestamps
3. **Trajectory CSV** (`_trajectory.csv`): Frame-by-frame coordinates

## Results JSON Structure

```json
{
  "video_path": "input.mp4",
  "output_path": "output.mp4", 
  "target_person": {
    "first_seen_frame": 123,
    "first_seen_timestamp": "00:04.100",
    "track_id": 5
  },
  "video_info": {
    "fps": 30.0,
    "frame_count": 900,
    "width": 1920,
    "height": 1080,
    "duration": 30.0
  }
}
```

## Troubleshooting

### Video won't open
- Ensure video file exists and is in supported format (MP4, AVI, MOV, MKV, WMV)
- Check file permissions

### No persons detected
- Lower confidence threshold: `--confidence 0.3`
- Check video quality and lighting
- Ensure persons are clearly visible

### Wrong person tracked
- Use manual selection: `--manual_select`
- Provide better reference image
- Adjust confidence threshold

### Poor tracking performance
- Use higher confidence: `--confidence 0.7`
- Ensure good video quality
- Avoid heavily crowded scenes

### Memory issues
- Process shorter video clips
- Reduce video resolution
- Close other applications

## Performance Tips

1. **Video Resolution**: HD (1920x1080) works best. 4K may be slow.
2. **Video Length**: Start with clips under 2 minutes for testing.
3. **Lighting**: Good lighting improves detection accuracy.
4. **Person Size**: Persons should be at least 50x50 pixels.
5. **Reference Images**: Use clear, well-lit photos of the target person.

## Technical Details

### Detection
- **Model**: YOLOv8 (You Only Look Once v8)
- **Confidence**: Minimum 0.5 recommended
- **Classes**: Detects "person" class only

### Tracking  
- **Algorithm**: DeepSORT (Simple Online Realtime Tracking with Deep Association Metric)
- **Features**: Motion prediction + visual similarity
- **Max Age**: 30 frames without detection before track deletion

### Visualization
- **Bounding Box**: Green for regular persons, Red for target
- **Flow Line**: Red trajectory with fading effect
- **Trail Length**: Last 50 points displayed
- **Smoothing**: Moving average over 5 frames

## Supported Formats

### Video Input
- MP4 (H.264/H.265)
- AVI  
- MOV
- MKV
- WMV

### Image Input (Reference)
- JPG/JPEG
- PNG
- BMP

### Video Output
- MP4 (H.264)

## System Requirements

- **Python**: 3.8 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: Optional but recommended for faster processing
- **Disk Space**: 2GB for installation + space for videos
- **OS**: Windows 10/11, macOS 10.14+, Ubuntu 18.04+

## Web Interface Features

The web interface provides:

1. **Drag & Drop Upload**: Easy video file upload
2. **Processing Options**: Configure detection and view settings
3. **Real-time Progress**: See processing status
4. **Automatic Download**: Get results when complete
5. **Mobile Friendly**: Works on tablets and phones

## API Usage (Advanced)

You can also use the tracker programmatically:

```python
from src.detector import PersonDetector
from src.tracker import PersonTracker
from src.visualizer import Visualizer

# Initialize components
detector = PersonDetector()
tracker = PersonTracker()  
visualizer = Visualizer()

# Process video frame by frame
for frame in video_frames:
    detections = detector.detect_persons(frame)
    tracked_objects = tracker.update(detections, frame)
    output_frame = visualizer.draw_all_tracks(frame, tracked_objects)
```

## Getting Help

If you encounter issues:

1. Check this usage guide
2. Look at the example scripts in `examples.py`
3. Review the troubleshooting section
4. Check the console output for error messages
5. Ensure all dependencies are properly installed

## License

This project is licensed under the MIT License. See LICENSE file for details.
