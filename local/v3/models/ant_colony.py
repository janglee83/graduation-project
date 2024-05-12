from pydantic import BaseModel
from typing import Optional, List, Any
from torch import Tensor, tensor
import torch
from helpers import FINISH_POINT_NAME, START_POINT_NAME
from models import ObjectHarmonySearch, HarmonySearch
from random import randint
from requests import TaskLinkageRequest


class AntColony(BaseModel):
    number_ants: int
    number_edge: int
    alpha: float
    beta: float
    best_ant_path: Optional[List]
    best_ant_path_length: Optional[float]
    relationship_kpi_matrix: Any
    default_pheromone_value: float
    pl: float
    pg: float
    pheromone_matrix: Optional[Any]
    duration_matrix: Any

    def __init__(self, number_ants: int, number_edge: int, relationship_kpi_matrix: Any, pheromone_matrix: Tensor, duration_matrix: Any, best_ant_path: List = [], best_ant_path_length: float = 0.0, alpha: float = 0.5, beta: float = 0.5, default_pheromone_value: float = 1.0, pl: float = 0.8, pg: float = 0.95) -> None:
        # pheromone_matrix = stack(pheromone_matrix)
        super().__init__(number_ants=number_ants, alpha=alpha, beta=beta, best_ant_path=best_ant_path, best_ant_path_length=best_ant_path_length,
                         number_edge=number_edge, relationship_kpi_matrix=relationship_kpi_matrix, default_pheromone_value=default_pheromone_value, pl=pl, pg=pg, pheromone_matrix=pheromone_matrix, duration_matrix=duration_matrix)

    def get_weight_base_rand_hms(self, harmony_search: HarmonySearch, row: int, col: int):
        # hms_layer_weight = harmony_search.harmony_memory[:, row, col]
        # pheromone_layer_value = self.pheromone_matrix[:, row, col]

        # total = sum(pow(hms_layer_weight, self.beta)
        #             * pow(pheromone_layer_value, self.alpha))

        # if total == 0:
        #     return tensor(0), tensor(0)

        # prob_trans_matrix = pow(hms_layer_weight, self.beta) * \
        #     pow(pheromone_layer_value, self.alpha) / total

        # _, max_index = max(prob_trans_matrix, dim=0)
        hms_rand = randint(0, harmony_search.objective_harmony_search.hms - 1)

        return hms_rand, harmony_search.harmony_memory[hms_rand, row, col]

    def get_weight_item_base_rand_hms(self, harmony_search: HarmonySearch, row: int, col: int, item: int):
        hms_layer_weight = harmony_search.harmony_memory[:, row, col, item]
        # pheromone_layer_value = self.pheromone_matrix[:, row, col, item]

        # total = sum(pow(hms_layer_weight, self.beta) *
        #             pow(pheromone_layer_value, self.alpha))

        # if total == 0:
        #     return tensor(0), tensor(0)

        # prob_trans_matrix = pow(hms_layer_weight, self.beta) * \
        #     pow(pheromone_layer_value, self.alpha) / total

        # _, max_index = torch.max(prob_trans_matrix, dim=0)

        hms_rand = randint(0, harmony_search.objective_harmony_search.hms - 1)

        return hms_rand, hms_layer_weight[hms_rand]

    def get_list_available_next_note(self, start_point: str, list_visited_point: list, num_task: int) -> list:
        if start_point == START_POINT_NAME:
            index_start = 0
        elif start_point == FINISH_POINT_NAME:
            index_start = num_task + 1
        else:
            index_start = int(start_point)

        visited_set = set(list_visited_point)
        kpi_matrix_tensor = self.relationship_kpi_matrix.clone().detach()
        row_tensor = kpi_matrix_tensor[index_start]
        indices = torch.nonzero(row_tensor > 0).squeeze()
        indices_list = indices.tolist()

        filtered_values = [FINISH_POINT_NAME if idx == num_task + 1 else str(idx)
                           for idx in indices_list if str(idx) not in visited_set]

        return filtered_values

    def get_path_weight(self, path_solution_candidates: Tensor, object_hs: ObjectHarmonySearch, from_edge: str, to_edge: str) -> Tensor:
        path_detail = next(
            (item for item in path_solution_candidates if item['from'] == from_edge and item['to'] == to_edge), None)

        weight_position = list()
        weight_values = list()
        for col in range(object_hs.number_parameters):
            position_depth, position_row, value = self.get_weight_base_rand_hms(
                path_detail['harmony_memory'][:, :, col].unsqueeze(-1), path_detail['harmony_pheromone_candidate_value'], col)
            weight_position.append([position_depth, position_row, col])
            weight_values.append(value)

        return tensor(weight_position), tensor(weight_values)

    def get_distance_point(self, harmony_search: HarmonySearch, to_edge: str, object_hs: ObjectHarmonySearch):
        kpi_index = int(to_edge) - 1

        weight_position = list()
        list_weight = list()
        for row in range(4):
            list_weight_item = list()
            for item in range(len(object_hs.human_score_vector)):

                position_harmony_memory, weight = self.get_weight_item_base_rand_hms(
                    harmony_search=harmony_search, row=row, col=kpi_index, item=item)

                list_weight_item.append(weight)
                weight_position.append(
                    [position_harmony_memory, row, kpi_index, item])

            list_weight.append(list_weight_item)

        return tensor(weight_position), tensor(list_weight)

    def calculate_prob_transit(self, harmony_search: HarmonySearch):
        probabilities = ((1 / torch.sum(harmony_search.harmony_memory, dim=3)) ** 0.9) * (
            self.pheromone_matrix ** 0.1) * ((1 / torch.sum(self.duration_matrix, dim=2)) ** (0.9))

        total = torch.sum(probabilities)
        prob_tensor = probabilities / total

        return prob_tensor

    def get_best_next_point(self, list_reachable_point: list, num_row: int, listTaskLinkage: list[TaskLinkageRequest], num_task: int, prob_transit: torch.Tensor):
        if not list_reachable_point:
            return FINISH_POINT_NAME, 0

        if list_reachable_point[-1] == FINISH_POINT_NAME and len(list_reachable_point) == 1:
            return FINISH_POINT_NAME, 0

        prob_point_tensor = prob_transit.sum(dim=(0))

        task_list = [START_POINT_NAME] + [str(i)
                                          for i in range(1, num_task + 1)] + [FINISH_POINT_NAME]
        list_reachable_point_set = set(list_reachable_point)
        task_list_set = set(task_list)
        list_not_reachable_point = list(
            task_list_set - list_reachable_point_set)

        if not (len(list_not_reachable_point) == 1 and list_not_reachable_point[-1] == FINISH_POINT_NAME):
            for point in list_not_reachable_point:
                if point != START_POINT_NAME:
                    task_id = int(point)

                    # Find corresponding KPI ID
                    kpi_id = next(
                        (task_linkage.kpi_metric_id for task_linkage in listTaskLinkage if task_linkage.task_id == task_id), None)

                    if kpi_id is not None:
                        row_index = (task_id - 1) % num_row
                        col_index = kpi_id - 1

                        prob_point_tensor[row_index, col_index] = torch.tensor(0.0)


            # Find the flattened index of the maximum value
            max_index = torch.argmax(prob_point_tensor)

            # Compute the position from the flattened index
            max_position = (
                max_index // prob_point_tensor.shape[1], max_index % prob_point_tensor.shape[1])

            selected_task_in_kpi, selected_kpi = max_position

            # Filter tasks in the selected KPI
            selected_kpi_tasks = filter(
                lambda task_linkage: task_linkage.kpi_metric_id == selected_kpi.item() + 1, listTaskLinkage)

            # Get the task ID directly
            point = next((str(task.task_id) for index, task in enumerate(
                selected_kpi_tasks) if index == selected_task_in_kpi), None)

            return point, torch.sum(self.duration_matrix[selected_task_in_kpi, selected_kpi, :])

        return FINISH_POINT_NAME, 0

    # def get_best_next_point(self, list_reachable_point: list, harmony_search: HarmonySearch, num_row: int, listTaskLinkage: list[TaskLinkageRequest], antWeight: torch.Tensor, num_task: int, prob_transit: torch.Tensor):
    #     if not list_reachable_point:
    #         return FINISH_POINT_NAME, [], 0

    #     if list_reachable_point[-1] == FINISH_POINT_NAME and len(list_reachable_point) == 1:
    #         return FINISH_POINT_NAME, [], 0

    #     prob_tensor = prob_transit

    #     task_list = [START_POINT_NAME] + \
    #         [str(i) for i in range(1, num_task + 1)] + [FINISH_POINT_NAME]
    #     list_reachable_point_set = set(list_reachable_point)
    #     task_list_set = set(task_list)
    #     list_not_reachable_point = task_list_set - list_reachable_point_set

    #     if not (len(list_not_reachable_point) == 1 and list_not_reachable_point[-1] == FINISH_POINT_NAME):
    #         mask = torch.ones_like(prob_tensor, dtype=torch.bool)
    #         for point in list_not_reachable_point:
    #             if point != START_POINT_NAME:
    #                 task_id = int(point)
    #                 kpi_id = next(
    #                     (task_linkage.kpi_metric_id for task_linkage in listTaskLinkage if task_linkage.task_id == task_id), None)
    #                 if kpi_id is not None:
    #                     row_index = (task_id - 1) % num_row
    #                     col_index = kpi_id - 1
    #                     mask[:, row_index, col_index] = 0
    #         prob_tensor = prob_tensor * mask

    #         max_position = torch.argmax(prob_tensor)
    #         selected_hms, selected_task_in_kpi, selected_kpi = torch.unravel_index(
    #             max_position, prob_tensor.shape)

    #         selected_kpi_tasks = filter(
    #             lambda task_linkage: task_linkage.kpi_metric_id == selected_kpi + 1, listTaskLinkage)
    #         point = next((str(task.task_id) for index, task in enumerate(
    #             selected_kpi_tasks) if index == selected_task_in_kpi), None)

    #         antWeight[selected_task_in_kpi, selected_kpi, :].copy_(
    #             harmony_search.harmony_memory[selected_hms, selected_task_in_kpi, selected_kpi, :])

    #         duration = torch.sum(
    #             self.duration_matrix[selected_task_in_kpi, selected_kpi, :])
    #         return point, torch.tensor([selected_hms, selected_task_in_kpi, selected_kpi]), duration

    #     return FINISH_POINT_NAME, [], 0


    # def calculate_fitness_base_path(self, ant_path: list, weight_position: Tensor, object_hs: ObjectHarmonySearch, path_solution_candidates: Tensor):
    #     path_length = list()
    #     for index, edge in enumerate(ant_path):
    #         if index + 1 < len(ant_path) - 1:
    #             path_detail = next(
    #                 (item for item in path_solution_candidates if item['from'] == edge and item['to'] == ant_path[index + 1]), None)
    #             harmony_memory = path_detail['harmony_memory'].clone().detach()
    #             item_position = weight_position[index].clone().detach()

    #             element = list()
    #             for depth, row, col in item_position:
    #                 element.append(
    #                     tensor(harmony_memory[depth.item(), row.item(), col.item()].clone().detach()))

    #             path_length.append(object_hs.get_fitness_tensor(
    #                 tensor(element), ant_path[index + 1]))

    #     return sum(stack(path_length))
