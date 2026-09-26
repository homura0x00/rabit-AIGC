from typing import Generic, Optional, TypeVar

from pydantic import BaseModel

# TODO data是任意類型
T = TypeVar("T")

class ApiResponse(BaseModel, Generic[T]):
    code: int = 0
    message: str = "success"
    data: Optional[T] = None