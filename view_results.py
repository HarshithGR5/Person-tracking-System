#!/usr/bin/env python3
"""
Simple viewer to display tracking results and open the output video.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def view_tracking_results(results_dir="output"):
    """View the latest tracking results"""
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        print(f"❌ Results directory '{results_dir}' not found!")
        return
    
    # Find the latest results file
    json_files = list(results_dir.glob("*_results.json"))
    if not json_files:
        print("❌ No results files found!")
        return
    
    # Get the most recent results file
    latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
    
    # Load and display results
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    print("🎬 PERSON TRACKING RESULTS")
    print("=" * 50)
    print(f"📹 Video: {results['video_path']}")
    print(f"📤 Output: {results['output_path']}")
    print(f"⏱️  Duration: {results['video_info']['duration']:.1f} seconds")
    print(f"🖼️  Resolution: {results['video_info']['width']}x{results['video_info']['height']}")
    print(f"🎞️  FPS: {results['video_info']['fps']:.1f}")
    print()
    
    if 'target_person' in results:
        target = results['target_person']
        print("🎯 TARGET PERSON:")
        print(f"   📍 Track ID: {target.get('track_id', 'N/A')}")
        print(f"   📅 First seen: Frame {target.get('first_seen_frame', 'N/A')}")
        print(f"   ⏰ Timestamp: {target.get('first_seen_timestamp', 'N/A')}")
    
    # Check if output video exists
    output_path = results_dir / results['output_path'].replace('output/', '')
    if output_path.exists():
        print(f"\n✅ Output video created: {output_path}")
        print(f"📊 File size: {output_path.stat().st_size / (1024*1024):.1f} MB")
        
        # Try to open the video with default player
        try:
            if sys.platform == "win32":
                os.startfile(str(output_path))
                print("🎬 Opening video with default player...")
            elif sys.platform == "darwin":
                subprocess.run(["open", str(output_path)])
                print("🎬 Opening video with default player...")
            else:
                subprocess.run(["xdg-open", str(output_path)])
                print("🎬 Opening video with default player...")
        except Exception as e:
            print(f"⚠️  Could not open video automatically: {e}")
            print(f"👆 Please open manually: {output_path}")
    else:
        print(f"\n❌ Output video not found: {output_path}")
    
    # Check trajectory file
    trajectory_file = output_path.with_suffix('.csv').name.replace('.mp4', '_trajectory.csv')
    trajectory_path = results_dir / trajectory_file
    if trajectory_path.exists():
        print(f"📈 Trajectory data: {trajectory_path}")
        
        # Count trajectory points
        with open(trajectory_path, 'r') as f:
            lines = f.readlines()
            point_count = len(lines) - 1  # Subtract header
        print(f"📊 Trajectory points: {point_count}")

if __name__ == "__main__":
    view_tracking_results()