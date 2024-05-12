from pydantic import BaseModel
from typing import Optional


class TaskLinkageRequest(BaseModel):
    kpi_metric_id: int
    task_id: int
    resource_ids: list
    duration_resource_ids: list
    task_weight: float
    resource_weight: Optional[list]

    def __init__(self, kpi_metric_id: int,
                 task_id: int,
                 resource_ids: list,
                 duration_resource_ids: list,
                 task_weight: float,
                 resource_weight: Optional[list] = None):
        super().__init__(kpi_metric_id=kpi_metric_id, task_id=task_id, resource_ids=resource_ids,
                         duration_resource_ids=duration_resource_ids, task_weight=task_weight, resource_weight=resource_weight)
