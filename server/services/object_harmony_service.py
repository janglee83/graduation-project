from typing import List
from requests import Employees, KpiRequest
from torch import tensor, Tensor
import torch


class ObjectHarmonyService(object):
    def build_employee_score_vector(listEmployees: List[Employees]) -> Tensor:
        return tensor(list(map(lambda employee: employee.point, listEmployees)))

    def build_kpi_value_vector(listKpis: List[KpiRequest]) -> Tensor:
        return tensor(list(map(lambda kpi: kpi.weight, listKpis)))

    def build_task_kpi_weight_vector(listKpis: List[KpiRequest]) -> Tensor:
        matrix = torch.zeros(3, len(listKpis))
        _ = list(map(lambda kpi, idx: matrix[:, int(kpi.id) - 1].copy_(torch.tensor(
            list(map(lambda task: task.weight, kpi.tasks)))), listKpis, range(len(listKpis))))

        return matrix
        # return tensor(list(map(lambda kpi: list(map(lambda task: task.weight, kpi.tasks)), listKpis)))
