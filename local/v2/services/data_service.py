from torch import zeros, int32, Tensor, stack
from models import HarmonySearch
import torch
from requests import TaskRequest, ResourceRequest, TaskLinkageRequest


class DataService:
    def __init__(self) -> None:
        pass

    def build_task_relationship(self, listTask: list[TaskRequest]) -> Tensor:
        matrix_size = len(listTask) + 2
        matrix = zeros(matrix_size, matrix_size, dtype=int32)

        for task in listTask:
            for pre_task in task.pre_tasks:
                matrix[pre_task, task.task_id] = 1

        matrix[0, :] = 1
        matrix[:, -1] = 1
        matrix[-1, :] = 0
        matrix[0, 0] = 0
        matrix[0, -1] = 0

        return matrix

    def build_duration_matrix(self, listTaskLinkage: list[TaskLinkageRequest], num_row: int, num_col: int, num_item: int) -> Tensor:
        matrix = zeros(num_row, num_col, num_item)

        for task_linkage in listTaskLinkage:
            for index, resource_id in enumerate(task_linkage.resource_ids):
                matrix[(task_linkage.task_id - 1) % num_row, task_linkage.kpi_metric_id - 1, resource_id - 1] = task_linkage.duration_resource_ids[index]

        return matrix

    def build_lower_upper_matrix(self, listResource: list[ResourceRequest], listTaskLinkage: list[TaskLinkageRequest]) -> Tensor:
        num_col = listTaskLinkage[-1].kpi_metric_id
        num_row = 3 # moi kpi anh xa tu 3 task
        num_depth = 2

        matrix = torch.zeros(num_row, num_col, num_depth)

        for task_linkage in listTaskLinkage:
            upper_lower_list_dict = list()
            for resource in listResource:
                for exp_task in resource.experience:
                    if exp_task['task_id'] == task_linkage.task_id:
                        upper_lower_list_dict.append(exp_task)
            min_lower_bound = min(upper_lower_list_dict, key=lambda x: x['lower_bound'])[
                'lower_bound'] / 5
            max_upper_bound = max(upper_lower_list_dict, key=lambda x: x['upper_bound'])[
                'upper_bound'] / 5 + 0.35
            matrix[(task_linkage.task_id - 1) % 3, task_linkage.kpi_metric_id -
                   1, :] = torch.tensor([min_lower_bound, max_upper_bound])

        return matrix

    # def build_executive_staff_matrix(listKpis: List[KpiRequest], listEmployees: List[EmployeeRequest]) -> Tensor:
    #     """
    #     Build executive staff matrix, col is kpi id, row is staff id. item is list kpi's task. If equal to 1, mean that that staff can do that kpi's task

    #     Args:
    #         listKpis (List[KpiRequest]): _description_
    #         listEmployees (List[Employees]): _description_

    #     Returns:
    #         Tensor: _description_
    #     """

    #     # TODO: Change 3 into list max task
    #     matrix = torch.zeros(3, len(listKpis), len(listEmployees))

    #     for kpi in listKpis:
    #         for task in kpi.tasks:
    #             task_index = int(task.id) - 1
    #             executed_staff = torch.zeros(len(listEmployees))

    #             for exec_staff in task.executive_staff:
    #                 if exec_staff == 'all':
    #                     executed_staff = torch.ones(len(listEmployees))
    #                 else:
    #                     executed_staff[int(exec_staff) - 1] = 1

    #             matrix[task_index, int(kpi.id) - 1] = executed_staff

    #     return matrix

    def build_hs_memory_candidate(self, harmony_search: HarmonySearch, lower_upper_matrix: Tensor, num_row: int, num_col: int, num_item: int) -> Tensor:
        object_hs = harmony_search.objective_harmony_search
        list_candidate = list(map(lambda _: harmony_search.initialize_harmony_memory(
            lower_upper_matrix, num_row, num_col, num_item), range(object_hs.hms)))

        return stack(list_candidate)
