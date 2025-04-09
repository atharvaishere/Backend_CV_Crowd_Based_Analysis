import os
import shutil
import zipfile
import io
from fastapi.responses import FileResponse, StreamingResponse
from typing import Optional

class FileService:
    def save_uploaded_file(self, file, destination: str) -> str:
        """Save uploaded file to destination path"""
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        with open(destination, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return destination

    def create_zip_response(self, files: dict, zip_name: str):
        """Create a zip file response from multiple files"""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for arcname, filepath in files.items():
                if os.path.exists(filepath):
                    if filepath.endswith(".zip"):
                        # Extract existing zip and add individual files
                        with zipfile.ZipFile(filepath) as existing_zip:
                            for name in existing_zip.namelist():
                                zip_file.writestr(
                                    f"{arcname}/{name}",
                                    existing_zip.read(name)
                                )
                    else:
                        zip_file.write(filepath, arcname=arcname)
        
        zip_buffer.seek(0)
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={zip_name}.zip"
            }
        )

    def get_file_response(self, filepath: str, media_type: str, filename: str):
        """Create a FileResponse for a single file"""
        if not os.path.exists(filepath):
            return None
        return FileResponse(
            filepath,
            media_type=media_type,
            filename=filename
        )