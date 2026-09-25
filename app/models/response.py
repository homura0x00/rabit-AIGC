from pydantic import BaseModel

# TODO data是任意類型
class Response(BaseModel):
    code: int
    data: list | None = None
    error: bool