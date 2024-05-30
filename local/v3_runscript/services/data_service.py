import torch
from requests import TaskAssignRequest, MetricRequest, TaskExpRequest, ResourceRequest, TaskRequest
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

    def build_lower_upper_tensor(self, listTaskExp: list[TaskExpRequest], listResources: list[ResourceRequest], listTaskAssigns: list[TaskAssignRequest], listMetrics: list[MetricRequest], num_row: int, num_col: int):
        num_depth = 2
        tensor = torch.zeros(num_row, num_col, num_depth)

        # Create dictionaries for faster lookups
        resource_dict = {resource.id: resource for resource in listResources}
        task_exp_dict = {
            task_exp.task_id: task_exp for task_exp in listTaskExp}

        # Precompute task to metric index mapping
        task_to_metric = {}
        for metric in listMetrics:
            for task_id in metric.list_of_task:
                task_to_metric[task_id] = metric.id - 1

        # Iterate through task assignments
        for task_assign in listTaskAssigns:
            task_id = task_assign.task_id
            task_index = (task_id - 1) % num_row
            metric_index = task_to_metric.get(task_id, None)

            if metric_index is not None:
                list_current_task_exp = task_exp_dict[task_id].list_exp

                # Create a dictionary for task_exp values
                task_exp_values = {exp.resource_id: exp.value /
                                   100 for exp in list_current_task_exp}

                list_score = []
                for assign_resource in task_assign.resource_ids:
                    task_exp_score = task_exp_values.get(assign_resource, 0)
                    human_score = resource_dict[assign_resource].score
                    list_score.append(task_exp_score * human_score)

                lower_bound = min(list_score) / len(task_assign.resource_ids)
                upper_bound = 1.5 if len(task_assign.resource_ids) == 1 else max(
                    list_score) / len(task_assign.resource_ids) + 0.5

                tensor[task_index, metric_index] = torch.tensor(
                    [lower_bound, upper_bound])

        return tensor

    def build_hs_memory_candidate(self, harmony_search: HarmonySearch, lower_upper_tensor: torch.Tensor, num_row: int, num_col: int, num_item: int, listTaskAssign: list[TaskAssignRequest]):
        object_hs = harmony_search.objective_harmony_search
        list_candidate = list(map(lambda _: harmony_search.initialize_harmony_memory(
            lower_upper_tensor, num_row, num_col, num_item, listTaskAssign), range(object_hs.hms)))

        return torch.stack(list_candidate)
