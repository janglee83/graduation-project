from torch import zeros, int32, Tensor, stack
from harmony_search import HarmonySearch
import torch
from task_request import TaskRequest
from resource_request import ResourceRequest
from task_linkage_request import TaskLinkageRequest


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
                matrix[(task_linkage.task_id - 1) % num_row, task_linkage.kpi_metric_id -
                       1, resource_id - 1] = task_linkage.duration_resource_ids[index]

        return matrix

    def build_lower_upper_matrix(self, listResource: list[ResourceRequest], listTaskLinkage: list[TaskLinkageRequest]) -> Tensor:
        num_col = listTaskLinkage[-1].kpi_metric_id
        num_row = 5  # moi kpi anh xa tu 5 task
        num_depth = 2

        matrix = torch.zeros(num_row, num_col, num_depth)

        for task_linkage in listTaskLinkage:
            upper_lower_list_dict = list()
            for resource in listResource:
                for exp_task in resource.experience:
                    if exp_task['task_id'] == task_linkage.task_id:
                        upper_lower_list_dict.append(exp_task)
            min_lower_bound = min(upper_lower_list_dict, key=lambda x: x['lower_bound'])[
                'lower_bound'] / 50
            max_upper_bound = max(upper_lower_list_dict, key=lambda x: x['upper_bound'])[
                'upper_bound'] / 50 + 0.035
            matrix[(task_linkage.task_id - 1) % num_row, task_linkage.kpi_metric_id -
                   1, :] = torch.tensor([min_lower_bound, max_upper_bound])

        return matrix

    def build_hs_memory_candidate(self, harmony_search: HarmonySearch, lower_upper_matrix: Tensor, num_row: int, num_col: int, num_item: int) -> Tensor:
        object_hs = harmony_search.objective_harmony_search
        list_candidate = list(map(lambda _: harmony_search.initialize_harmony_memory(
            lower_upper_matrix, num_row, num_col, num_item), range(object_hs.hms)))

        return stack(list_candidate)
