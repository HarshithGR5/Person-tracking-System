"""
Flask web application for person tracking with file upload interface.
Provides a user-friendly web interface for video upload and processing.
"""

from flask import Flask, render_template, request, jsonify, send_file, url_for
from flask_cors import CORS
import os
import tempfile
import json
import subprocess
import threading
from datetime import datetime
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'uploads')
OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), '..', 'output')
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Global storage for processing jobs
processing_jobs = {}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE


def allowed_video_file(filename):
    """Check if video file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


def allowed_image_file(filename):
    """Check if image file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


class ProcessingJob:
    """Class to track processing job status."""
    def __init__(self, job_id, video_path, options, ref_image_path=None):
        self.job_id = job_id
        self.video_path = video_path
        self.ref_image_path = ref_image_path
        self.options = options
        self.status = 'queued'
        self.progress = 0
        self.result_path = None
        self.error_message = None
        self.created_at = datetime.now()
        self.started_at = None
        self.finished_at = None


def run_tracking_process(job):
    """Run the tracking process in background."""
    try:
        job.status = 'processing'
        job.started_at = datetime.now()
        
        # Build command - use virtual environment Python
        project_root = os.path.dirname(os.path.dirname(__file__))  # person_tracker directory
        venv_python = os.path.join(project_root, '..', '.venv', 'Scripts', 'python.exe')
        main_script = os.path.join(project_root, 'src', 'main.py')
        
        if not os.path.exists(venv_python):
            venv_python = 'python'  # fallback to system python
            
        cmd = [
            venv_python, 
            main_script,
            '--video', job.video_path,
            '--view', job.options.get('view_mode', 'flow')
        ]
        
        # Add reference image if provided
        if job.ref_image_path:
            cmd.extend(['--ref_image', job.ref_image_path])
        
        # Add optional arguments
        if job.options.get('confidence'):
            cmd.extend(['--confidence', str(job.options['confidence'])])
            
        if job.options.get('manual_select') and not job.ref_image_path:
            cmd.append('--manual_select')
            
        # Generate output path - match input video format
        input_ext = os.path.splitext(job.video_path)[1]
        if not input_ext:
            input_ext = '.mp4'  # Default fallback
        output_filename = f"tracked_{job.job_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{input_ext}"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        cmd.extend(['--output', output_path])
        
        # Run process from the correct working directory
        working_dir = os.path.dirname(os.path.dirname(__file__))  # person_tracker directory
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=working_dir)
        stdout, stderr = process.communicate()
        
        # Log detailed output for debugging
        print(f"Command: {' '.join(cmd)}")
        print(f"Return code: {process.returncode}")
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")
        
        if process.returncode == 0:
            job.status = 'completed'
            job.result_path = output_path
        else:
            job.status = 'failed'
            job.error_message = f"Command failed with code {process.returncode}. STDERR: {stderr}. STDOUT: {stdout}"
            
    except Exception as e:
        job.status = 'failed'
        job.error_message = str(e)
    
    finally:
        job.finished_at = datetime.now()


@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video and reference image upload."""
    try:
        # Validate video file
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No video file selected'}), 400
        
        if not allowed_video_file(video_file.filename):
            return jsonify({'error': 'Invalid video file type. Allowed: mp4, avi, mov, mkv, wmv'}), 400
        
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Save video file
        video_filename = secure_filename(f"{job_id}_video_{video_file.filename}")
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        video_file.save(video_path)
        
        # Handle optional reference image
        ref_image_path = None
        if 'reference_image' in request.files:
            ref_image_file = request.files['reference_image']
            if ref_image_file.filename != '' and allowed_image_file(ref_image_file.filename):
                ref_image_filename = secure_filename(f"{job_id}_ref_{ref_image_file.filename}")
                ref_image_path = os.path.join(app.config['UPLOAD_FOLDER'], ref_image_filename)
                ref_image_file.save(ref_image_path)
        
        # Get processing options
        options = {
            'view_mode': request.form.get('view_mode', 'flow'),
            'confidence': float(request.form.get('confidence', 0.5)),
            'manual_select': request.form.get('manual_select') == 'true' and ref_image_path is None
        }
        
        # Create processing job
        job = ProcessingJob(job_id, video_path, options, ref_image_path)
        processing_jobs[job_id] = job
        
        # Start processing in background
        thread = threading.Thread(target=run_tracking_process, args=(job,))
        thread.daemon = True
        thread.start()
        
        message = 'Video uploaded successfully. Processing started.'
        if ref_image_path:
            message += ' Using reference image for automatic person detection.'
        
        return jsonify({
            'job_id': job_id,
            'status': 'queued',
            'message': message
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/status/<job_id>')
def get_status(job_id):
    """Get processing job status."""
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = processing_jobs[job_id]
    
    response = {
        'job_id': job_id,
        'status': job.status,
        'progress': job.progress,
        'created_at': job.created_at.isoformat(),
        'started_at': job.started_at.isoformat() if job.started_at else None,
        'finished_at': job.finished_at.isoformat() if job.finished_at else None
    }
    
    if job.status == 'completed' and job.result_path:
        response['download_url'] = url_for('download_result', job_id=job_id)
    
    if job.status == 'failed' and job.error_message:
        response['error_message'] = job.error_message
    
    return jsonify(response)


@app.route('/download/<job_id>')
def download_result(job_id):
    """Download processed video."""
    if job_id not in processing_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = processing_jobs[job_id]
    
    if job.status != 'completed' or not job.result_path or not os.path.exists(job.result_path):
        return jsonify({'error': 'Result not available'}), 404
    
    return send_file(job.result_path, as_attachment=True, 
                    download_name=f'tracked_video_{job_id}.mp4')


@app.route('/jobs')
def list_jobs():
    """List all processing jobs."""
    jobs_info = []
    for job_id, job in processing_jobs.items():
        jobs_info.append({
            'job_id': job_id,
            'status': job.status,
            'created_at': job.created_at.isoformat(),
            'options': job.options
        })
    
    return jsonify(jobs_info)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
