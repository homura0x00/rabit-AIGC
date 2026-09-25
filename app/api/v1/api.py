from pathlib import Path
import uuid

from fastapi import APIRouter, HTTPException, UploadFile
from app.core.log import logger

# UPLOAD_DIR = Path("./uploads")
# UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
FILE_TYPE = {"application/pdf"}
MAX_SIZE: int = 10 * 1024 * 1024 # 10MB

api_router = APIRouter()

async def health_check():
    """Health check endpoint.
    
    Returns:
        dict: Health status information.
    """
    logger.info("health_check_called")
    return {"status": "healthy", "version": "1.0.0"}


@api_router.post("/resumes")
async def create_upload_file(file: UploadFile):
    """临时存储，写入数据库后删除"""
    if not file:
        return {"error": "No file uploaded."}
    
    if file.content_type not in FILE_TYPE:
        return {"error": "Invalid file type. Only PDF files are allowed."}

    # safe_filename = f"{uuid.uuid4().hex}_{file.filename}"
    # file_path = f"{UPLOAD_DIR}/{safe_filename}.txt"
    # with open(file_path, "wb") as f:
    #     f.write(await file.read())

    data = await file.read()

    if len(data) > MAX_SIZE:
        del data
        raise HTTPException(413, "文件过大")

    del data

    logger.info(f"File uploaded: {file.filename}")
    return {"status": "ok", "filename": file.filename, "file_size": file.size}