from pydantic import BaseModel


class ResourceRequest(BaseModel):
    id: int
    type: str
    code: str
    name: str
    description: str
    certificates: list
    major: str
    major_type: str
    score: float
    # experience: list
