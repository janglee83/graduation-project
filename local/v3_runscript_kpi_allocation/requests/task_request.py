from pydantic import BaseModel


class TaskRequest(BaseModel):
    task_id: int
    description: str
    # pre_tasks: list
    metric_id: int
    value: float
    unit: str
    weight: float
    start_date: str
    end_date: str
    duration: float


class TaskAssignRequest(BaseModel):
    task_id: int
    resource_ids: list[int]
    metric_id: int


class ExpInstance(BaseModel):
    resource_id: int
    value: float


class TaskExpRequest(BaseModel):
    task_id: int
    list_exp: list[ExpInstance]
