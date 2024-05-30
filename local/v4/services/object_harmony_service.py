from torch import Tensor
import torch
from requests import TaskLinkageRequest


class ObjectHarmonyService(object):
    def build_task_kpi_weight_vector(listTaskLinkage: list[TaskLinkageRequest], numberKpi: int, numberTaskEach: int) -> Tensor:
        matrix = torch.zeros(numberTaskEach, numberKpi)

        for task_linkage in listTaskLinkage:
            matrix[(task_linkage.task_id - 1) % numberTaskEach,
                   task_linkage.kpi_metric_id - 1] = torch.tensor(task_linkage.task_weight)

        return matrix
