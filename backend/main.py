from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
import shutil
import os
import uuid
import atexit
import subprocess
from typing import Optional, Dict
from pathlib import Path

from backend.models.job_manager import AnalysisJobManager
from backend.models.schemas import AnalysisRequest, AnalysisResponse
from backend.services.video_service import VideoProcessor
from backend.services.file_service import FileService
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="API",
    description="API for Crowd Scene interactions in videos"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Your Vite frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Length"]
)

# Initialize services
job_manager = AnalysisJobManager()
video_processor = VideoProcessor()
file_service = FileService()

# Configuration
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.mov', '.avi', '.mkv']

def encode_video_for_web(input_path: str, output_path: str) -> bool:
    """Encode video for web playback using FFmpeg"""
    try:
        cmd = [
            'ffmpeg',
            '-i', input_path,
            '-c:v', 'libx264',  # H.264 codec
            '-profile:v', 'main',
            '-preset', 'fast',
            '-movflags', '+faststart',  # Enable streaming
            '-pix_fmt', 'yuv420p',  # Standard pixel format for compatibility
            '-crf', '23',  # Quality balance (18-28, lower is better)
            '-c:a', 'aac',  # Audio codec
            '-b:a', '128k',  # Audio bitrate
            '-y',  # Overwrite output file if exists
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg encoding failed: {e.stderr.decode()}")
        return False
    except Exception as e:
        print(f"Video encoding error: {str(e)}")
        return False

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_behavior(
    video: UploadFile = File(...),
    output_folder: Optional[str] = "./output",
    desired_fps: Optional[int] = 10,
    confidence_threshold: Optional[float] = 0.3,
    iou_threshold: Optional[float] = 0.4
):
    """Endpoint to upload and analyze a video"""
    # File size validation
    await validate_file_size(video)
    
    # Validate file extension
    file_ext = Path(video.filename).suffix.lower()
    if file_ext not in SUPPORTED_VIDEO_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported formats: {', '.join(SUPPORTED_VIDEO_FORMATS)}"
        )
    
    job_id = job_manager.create_job(video.filename, os.path.join(output_folder, str(uuid.uuid4())))
    
    try:
        # Save uploaded video
        video_path = os.path.join(job_manager.jobs[job_id]["output_folder"], video.filename)
        file_service.save_uploaded_file(video, video_path)

        # Process video (behavior analysis)
        results = video_processor.process_video(
            video_path=video_path,
            output_folder=job_manager.jobs[job_id]["output_folder"],
            desired_fps=desired_fps,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold
        )

        # Encode the analyzed video for web playback
        analyzed_video_path = results["output_video_path"]
        web_video_path = os.path.join(job_manager.jobs[job_id]["output_folder"], f"web_{Path(analyzed_video_path).name}")
        
        if not encode_video_for_web(analyzed_video_path, web_video_path):
            raise HTTPException(status_code=500, detail="Video encoding failed")
        
        # Update results with web-optimized video path
        results["output_video_path"] = web_video_path

        # Register cleanup
        register_cleanup([
            video_path,
            analyzed_video_path,
            web_video_path,
            results["report_path"],
            results["plots_path"],
            job_manager.jobs[job_id]["output_folder"]
        ])

        # Update job status
        job_manager.update_job(job_id, {
            "status": "completed",
            "output_video_path": web_video_path,
            "report_path": results["report_path"],
            "plots_path": results["plots_path"]
        })

        return job_manager.get_job(job_id)

    except Exception as e:
        job_manager.update_job(job_id, {
            "status": "failed",
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-and-return-video")
async def analyze_and_return_video(
    video: UploadFile = File(...),
    output_folder: Optional[str] = "./output",
    desired_fps: Optional[int] = 10,
    confidence_threshold: Optional[float] = 0.3,
    iou_threshold: Optional[float] = 0.4
):
    """Endpoint that ONLY returns the analyzed video file"""
    # File size validation
    await validate_file_size(video)
    
    # Validate file extension
    file_ext = Path(video.filename).suffix.lower()
    if file_ext not in SUPPORTED_VIDEO_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported formats: {', '.join(SUPPORTED_VIDEO_FORMATS)}"
        )
    
    try:
        # Create job directory
        job_id = str(uuid.uuid4())
        job_folder = os.path.join(output_folder, job_id)
        os.makedirs(job_folder, exist_ok=True)
        
        # Save uploaded video
        video_path = os.path.join(job_folder, video.filename)
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # Process video (behavior analysis)
        results = video_processor.process_video(
            video_path=video_path,
            output_folder=job_folder,
            desired_fps=desired_fps,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold
        )
        
        # Encode the analyzed video for web playback
        analyzed_video_path = results["output_video_path"]
        web_video_path = os.path.join(job_folder, f"web_{Path(analyzed_video_path).name}")
        
        if not encode_video_for_web(analyzed_video_path, web_video_path):
            raise HTTPException(status_code=500, detail="Video encoding failed")
        
        # Get the file size for headers
        file_size = os.path.getsize(web_video_path)
        
        # Register cleanup for temporary files
        register_cleanup([video_path, analyzed_video_path, web_video_path, job_folder])
        
        # Return ONLY the video file
        return FileResponse(
            web_video_path,
            media_type="video/mp4",
            filename="analyzed_video.mp4",
            headers={
                "Content-Length": str(file_size),
                "Accept-Ranges": "bytes",
                "Content-Disposition": "inline",
                "Access-Control-Expose-Headers": "Content-Length"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions
async def validate_file_size(file: UploadFile):
    """Validate that the file size is within limits"""
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset pointer
    
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {MAX_FILE_SIZE/1024/1024}MB"
        )

def register_cleanup(files_to_clean: list):
    """Register files/folders for automatic cleanup"""
    def cleanup():
        for path in files_to_clean:
            try:
                if os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
            except:
                pass
    
    atexit.register(cleanup)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)