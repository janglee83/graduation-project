from torch import tensor, Tensor
import torch
from requests import TaskLinkageRequest


class ObjectHarmonyService(object):
    # def build_employee_score_vector(listEmployees: List[EmployeeRequest]) -> Tensor:
    #     return tensor(list(map(lambda employee: employee.score * employee.task_completion_rate, listEmployees)))

    # def build_kpi_value_vector(listKpis: List[KpiRequest]) -> Tensor:
    #     return tensor(list(map(lambda kpi: kpi.weight, listKpis)))

    def build_task_kpi_weight_vector(listTaskLinkage: list[TaskLinkageRequest], numberKpi: int, numberTaskEach: int) -> Tensor:
        matrix = torch.zeros(numberTaskEach, numberKpi)

        for task_linkage in listTaskLinkage:
            matrix[(task_linkage.task_id - 1) % numberTaskEach, task_linkage.kpi_metric_id - 1] = torch.tensor(task_linkage.task_weight)

        return matrix
        # _ = list(map(lambda kpi, idx: matrix[:, int(kpi.id) - 1].copy_(torch.tensor(
        #     list(map(lambda task: task.weight, kpi.tasks)))), listKpis, range(len(listKpis))))

        # return matrix
        # return tensor(list(map(lambda kpi: list(map(lambda task: task.weight, kpi.tasks)), listKpis)))
