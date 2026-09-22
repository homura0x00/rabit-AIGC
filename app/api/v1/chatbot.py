
from fastapi import APIRouter, Request


router = APIRouter()

@router.post("/chat")
async def chat(
    request: Request,
):
    pass