from typing import Dict
from uuid import uuid4
from backend.models.schemas import AnalysisResponse

class AnalysisJobManager:
    def __init__(self):
        self.jobs: Dict[str, dict] = {}

    def create_job(self, video_path: str, output_folder: str) -> str:
        job_id = str(uuid4())
        self.jobs[job_id] = {
            "status": "processing",
            "video_path": video_path,
            "output_folder": output_folder
        }
        return job_id

    def update_job(self, job_id: str, updates: dict) -> None:
        if job_id in self.jobs:
            self.jobs[job_id].update(updates)

    def get_job(self, job_id: str) -> AnalysisResponse:
        if job_id not in self.jobs:
            return None
        
        job = self.jobs[job_id]
        return AnalysisResponse(
            job_id=job_id,
            status=job["status"],
            output_video_path=job.get("output_video_path"),
            report_path=job.get("report_path"),
            plots_path=job.get("plots_path"),
            error=job.get("error")
        )