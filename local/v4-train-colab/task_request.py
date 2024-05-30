from pydantic import BaseModel


class TaskRequest(BaseModel):
    task_id: int
    task_type: str
    description: str
    pre_tasks: list
    requirements: dict
    affected_factors: list
