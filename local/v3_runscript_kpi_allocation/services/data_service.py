import torch
from requests import TaskAssignRequest, MetricRequest, TaskExpRequest, ResourceRequest, TaskRequest, UnitRequestResource
from models import HarmonySearch


class DataService:
    def __init__(self):
        pass

    def find_number_row(self, listMetrics: list[MetricRequest]):
        return max(len(metric.list_of_task) for metric in listMetrics)

    def find_number_item(self, listTaskAssign: list[TaskAssignRequest]):
        return max(len(task_assign.resource_ids) for task_assign in listTaskAssign)

    # def build_task_relationship(self, listTask: list[TaskRequest]) -> Tensor:
    #     matrix_size = len(listTask) + 2
    #     matrix = zeros(matrix_size, matrix_size, dtype=int32)

    #     for task in listTask:
    #         for pre_task in task.pre_tasks:
    #             matrix[pre_task, task.task_id] = 1

    #     matrix[0, :] = 1
    #     matrix[:, -1] = 1
    #     matrix[-1, :] = 0
    #     matrix[0, 0] = 0
    #     matrix[0, -1] = 0

    #     return matrix

    def build_duration_matrix(self, listTasks: list[TaskRequest], num_row: int, num_col: int):
        matrix = torch.zeros(num_row, num_col)

        for task in listTasks:
            task_index = (task.task_id - 1) % num_row
            metric_index = task.metric_id - 1
            matrix[task_index, metric_index] = torch.tensor(task.duration)

        return matrix

    def build_lower_upper_tensor(self, num_row: int, listUnitResource: list[UnitRequestResource]):
        num_depth = 2
        tensor = torch.zeros(num_row, num_depth)

        list_lower = list()
        list_upper = list()
        for row in range(num_row):
            min_score_scaled = listUnitResource[row].min_score / num_row / 100
            max_score_scaled = listUnitResource[row].max_score / num_row / 100 + 1
            list_lower.append(min_score_scaled)
            list_upper.append(max_score_scaled)

        min_val = min(list_lower)
        max_val = max(list_upper)

        for row in range(num_row):
            tensor[row, 0] = min_val
            tensor[row, 1] = max_val

        return tensor

    def build_hs_memory_candidate(self, harmony_search: HarmonySearch, lower_upper_tensor: torch.Tensor, num_row: int,
                                  num_col: int):
        object_hs = harmony_search.objective_harmony_search
        list_candidate = list(map(lambda _: harmony_search.initialize_harmony_memory(
            lower_upper_tensor, num_row, num_col), range(object_hs.hms)))

        return torch.stack(list_candidate)
