"""API v1 router aggregation.

Routers are collected here rather than in ``main.py`` so the application entry
point stays about application lifecycle. Adding a feature module is one line.
"""

from fastapi import APIRouter

from app.api.v1.api import router as resume_router
from app.api.v1.chatbot import router as chat_router
from app.api.v1.jobs import router as job_router
from app.api.v1.screening import router as screening_router

api_router = APIRouter()
api_router.include_router(resume_router, tags=["resumes"])
api_router.include_router(job_router, tags=["jobs"])
api_router.include_router(screening_router, tags=["screening"])
api_router.include_router(chat_router, tags=["chat"])

__all__ = ["api_router"]
