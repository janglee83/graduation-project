from pydantic import BaseModel


class EnvironmentTaskScore(BaseModel):
    task_id: int
    mean_score: float


class ProductTaskScore(BaseModel):
    task_id: int
    mean_score: float
