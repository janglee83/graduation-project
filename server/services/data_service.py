from typing import List
from requests import KpiConditionRequest, KpiRequest, Employees, TaskRequest
from torch import zeros, int32, Tensor, tensor, stack, flip, transpose
from models import HarmonySearch
import torch
# from helpers import START_POINT_NAME, FINISH_POINT_NAME
# from models import ObjectHarmonySearch
# from services.truncated_normal_service import TruncatedNormalService


class DataService:
    def build_kpi_relationship_matrix(kpiConditions: List[KpiConditionRequest]) -> Tensor:
        """
        Build a KPI relationship matrix base post condition of each kpi.

        Args:
            kpiConditions (List[KpiConditionRequest]): List of KPI condition requests including kpi_id and its post condition.

        Returns:
            Tensor: The KPI relationship matrix.
        """
        matrix_size = len(kpiConditions) + 2
        matrix = zeros(matrix_size, matrix_size, dtype=int32)

        for kpi_condition in kpiConditions:
            index_kpi_col = int(kpi_condition.id)
            if index_kpi_col >= matrix_size:
                continue

            if matrix[0, index_kpi_col] == 0:
                matrix[0, index_kpi_col] = 1

            for next_point in kpi_condition.post_condition:
                next_point_index = int(next_point)
                if next_point_index >= matrix_size:
                    continue
                matrix[index_kpi_col, next_point_index] = 1

        return matrix

    def build_lower_upper_matrix(listKpis: List[KpiRequest]) -> Tensor:
        """
        Build a lower_upper_matrix base lower and upper bound's value of each kpi's task

        Args:
            listKpis (List[KpiRequest]): List of Kpi's detail

        Returns:
            Tensor: lower_upper_matrix
        """
        matrix = torch.zeros(3, len(listKpis), 2)

        _ = list(map(lambda kpi, idx: matrix[:, int(kpi.id) - 1].copy_(torch.tensor(
            list(map(lambda task: (task.lower_bound, task.upper_bound), kpi.tasks)))), listKpis, range(len(listKpis))))

        return matrix

        # return tensor(list(map(lambda kpi: list(map(lambda task: (task.lower_bound, task.upper_bound), kpi.tasks)), listKpis)))

    def build_executive_staff_matrix(listKpis: List[KpiRequest], listEmployees: List[Employees]) -> Tensor:
        """
        Build executive staff matrix, col is kpi id, row is staff id. item is list kpi's task. If equal to 1, mean that that staff can do that kpi's task

        Args:
            listKpis (List[KpiRequest]): _description_
            listEmployees (List[Employees]): _description_

        Returns:
            Tensor: _description_
        """

        # TODO: Change 3 into list max task
        matrix = torch.zeros(3, len(listKpis), len(listEmployees))

        for kpi in listKpis:
            for task in kpi.tasks:
                task_index = int(task.id) - 1
                executed_staff = torch.zeros(len(listEmployees))

                for exec_staff in task.executive_staff:
                    if exec_staff == 'all':
                        executed_staff = torch.ones(len(listEmployees))
                    else:
                        executed_staff[int(exec_staff) - 1] = 1

                matrix[task_index, int(kpi.id) - 1] = executed_staff

        return matrix

    # def build_executive_task_base_kpi_matrix(listKpis: List[KpiRequest], listTasks: List[TaskRequest]) -> Tensor:
    #     """Build executive task base kpi matrix, row is num task, col is num kpi

    #     Args:
    #         listKpis (List[KpiRequest]): _description_

    #     Returns:
    #         Tensor: _description_
    #     """
    #     # return tensor(list(map(lambda kpi: kpi.task_weight, listKpis)))
    #     row_len = len(listTasks)
    #     list_task_weights = [kpi.task_weight for kpi in listKpis]
    #     return tensor([[task_weight[i] for task_weight in list_task_weights] for i in range(row_len)])

    def build_hs_memory_candidate(harmony_search: HarmonySearch, lower_upper_matrix: Tensor, executive_task_staff_matrix: Tensor) -> Tensor:
        object_hs = harmony_search.objective_harmony_search
        list_candidate = list(map(lambda _: harmony_search.initialize_harmony_memory(
            lower_upper_matrix, executive_task_staff_matrix), range(object_hs.hms)))

        return stack(list_candidate)
