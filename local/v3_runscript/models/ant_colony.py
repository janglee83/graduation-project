from pydantic import BaseModel
from typing import Optional, List, Any
from torch import Tensor
import torch
from helpers import FINISH_POINT_NAME, START_POINT_NAME


class AntColony(BaseModel):
    number_ants: int
    alpha: float
    beta: float
    default_pheromone_value: float
    pheromone_matrix: Optional[Any]
    duration_matrix: Any

    def __init__(self, number_ants: int, pheromone_matrix: Tensor, duration_matrix: Any, alpha: float = 0.5, beta: float = 0.5, default_pheromone_value: float = 1.0) -> None:
        super().__init__(number_ants=number_ants, alpha=alpha, beta=beta, default_pheromone_value=default_pheromone_value,
                         pheromone_matrix=pheromone_matrix, duration_matrix=duration_matrix)

    # use
    # def get_list_available_next_note(self, start_point: str, list_visited_point: list, num_task: int) -> list:
    #     if start_point == START_POINT_NAME:
    #         index_start = 0
    #     elif start_point == FINISH_POINT_NAME:
    #         index_start = num_task + 1
    #     else:
    #         index_start = int(start_point)

    #     visited_set = set(list_visited_point)
    #     kpi_matrix_tensor = self.relationship_kpi_matrix.clone().detach()
    #     row_tensor = kpi_matrix_tensor[index_start]
    #     indices = torch.nonzero(row_tensor > 0).squeeze()
    #     indices_list = indices.tolist()

    #     filtered_values = [FINISH_POINT_NAME if idx == num_task + 1 else str(idx)
    #                        for idx in indices_list if str(idx) not in visited_set]

    #     return filtered_values

    # # use
    # def get_best_next_point(self, list_reachable_point: list, num_row: int, listTaskLinkage: list[TaskLinkageRequest], num_task: int, prob_transit: torch.Tensor):
    #     if not list_reachable_point:
    #         return FINISH_POINT_NAME, 0

    #     if list_reachable_point[-1] == FINISH_POINT_NAME and len(list_reachable_point) == 1:
    #         return FINISH_POINT_NAME, 0

    #     prob_point_tensor = prob_transit.sum(dim=(0))

    #     task_list = [START_POINT_NAME] + [str(i)
    #                                       for i in range(1, num_task + 1)] + [FINISH_POINT_NAME]
    #     list_reachable_point_set = set(list_reachable_point)
    #     task_list_set = set(task_list)
    #     list_not_reachable_point = list(
    #         task_list_set - list_reachable_point_set)

    #     if not (len(list_not_reachable_point) == 1 and list_not_reachable_point[-1] == FINISH_POINT_NAME):
    #         for point in list_not_reachable_point:
    #             if point != START_POINT_NAME:
    #                 task_id = int(point)

    #                 # Find corresponding KPI ID
    #                 kpi_id = next(
    #                     (task_linkage.kpi_metric_id for task_linkage in listTaskLinkage if task_linkage.task_id == task_id), None)

    #                 if kpi_id is not None:
    #                     row_index = (task_id - 1) % num_row
    #                     col_index = kpi_id - 1

    #                     prob_point_tensor[row_index,
    #                                       col_index] = torch.tensor(0.0)

    #         # Find the flattened index of the maximum value
    #         max_index = torch.argmax(prob_point_tensor)

    #         # Compute the position from the flattened index
    #         max_position = (
    #             max_index // prob_point_tensor.shape[1], max_index % prob_point_tensor.shape[1])

    #         selected_task_in_kpi, selected_kpi = max_position

    #         # Filter tasks in the selected KPI
    #         selected_kpi_tasks = filter(
    #             lambda task_linkage: task_linkage.kpi_metric_id == selected_kpi.item() + 1, listTaskLinkage)

    #         # Get the task ID directly
    #         point = next((str(task.task_id) for index, task in enumerate(
    #             selected_kpi_tasks) if index == selected_task_in_kpi), None)

    #         return point, torch.sum(self.duration_matrix[selected_task_in_kpi, selected_kpi, :])

    #     return FINISH_POINT_NAME, 0
