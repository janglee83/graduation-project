from pydantic import BaseModel
from typing import List

class TaskResponse(BaseModel):
    task_id: str
    list_staff_weight: List[float]

class CoreResponse(BaseModel):
    kpi_id: str
    tasks: List[TaskResponse]
    fitness_value: float

