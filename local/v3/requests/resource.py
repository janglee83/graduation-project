from pydantic import BaseModel


class ResourceRequest(BaseModel):
    id: int
    type: str
    description: str
    qualifications: dict
    experience: list
