from pydantic import BaseModel
from typing import List, Optional


class TaskStaffWeight(BaseModel):
    staff_id: str
    weight: float


class TaskResponse(BaseModel):
    task_id: str
    list_staff_weight: Optional[List[TaskStaffWeight]]


class CoreResponse(BaseModel):
    kpi_id: str
    tasks: List[TaskResponse]
    fitness_value: float
