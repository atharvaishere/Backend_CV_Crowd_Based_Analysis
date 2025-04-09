from pydantic import BaseModel
from typing import Optional

class AnalysisRequest(BaseModel):
    video_path: str
    output_folder: Optional[str] = "./output"
    desired_fps: Optional[int] = 10
    confidence_threshold: Optional[float] = 0.3
    iou_threshold: Optional[float] = 0.4

class AnalysisResponse(BaseModel):
    job_id: str
    status: str
    output_video_path: Optional[str]
    report_path: Optional[str]
    plots_path: Optional[str]
    error: Optional[str]