from pydantic import BaseModel
from typing import Optional, List, Any
from torch import Tensor, pow, sum, max, nonzero, tensor, stack
from helpers import convert_edge_str_to_index
from helpers import FINISH_POINT_NAME
from models import ObjectHarmonySearch, HarmonySearch
from random import randint


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

    def __init__(self, number_ants: int, number_edge: int, relationship_kpi_matrix: Any, pheromone_matrix: Tensor, best_ant_path: List = [], best_ant_path_length: float = 0.0, alpha: float = 0.4, beta: float = 0.6, default_pheromone_value: float = 1.0, pl: float = 0.4, pg: float = 0.6) -> None:
        # pheromone_matrix = stack(pheromone_matrix)
        super().__init__(number_ants=number_ants, alpha=alpha, beta=beta, best_ant_path=best_ant_path, best_ant_path_length=best_ant_path_length,
                         number_edge=number_edge, relationship_kpi_matrix=relationship_kpi_matrix, default_pheromone_value=default_pheromone_value, pl=pl, pg=pg, pheromone_matrix=pheromone_matrix)

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

    def get_list_available_next_note(self, start_point: str, list_visited_point: list) -> list:
        index_start_edge = convert_edge_str_to_index(
            start_point, self.number_edge)

        indices = nonzero(self.relationship_kpi_matrix[index_start_edge] > 0)

        filtered_values = [str(item[0].item())
                           for item in indices if str(item[0].item()) not in list_visited_point]

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

        while True:
            weight_position = list()
            list_weight = list()
            for row in range(len(harmony_search.objective_harmony_search.human_score_vector)):
                position_harmony_memory, weight = self.get_weight_base_rand_hms(
                    harmony_search=harmony_search, row=row, col=kpi_index)

                weight_position.append(
                    [position_harmony_memory, row, kpi_index])
                list_weight.append(weight)

            fitness = object_hs.get_fitness_base_kpi(
                stack(list_weight), kpi_index=kpi_index)

            if fitness >= 0:
                break

        return tensor(weight_position), tensor(list_weight)

    def get_best_next_point(self, start_point: str, list_reachable_point: list, harmony_search: HarmonySearch, ant_weight: Tensor):
        if not list_reachable_point:
            return FINISH_POINT_NAME, [], []

        object_hs = harmony_search.objective_harmony_search

        list_point_weight_position, list_point_weight = zip(
            *map(lambda point: self.get_distance_point(harmony_search=harmony_search, to_edge=int(point), object_hs=object_hs), list_reachable_point))

        list_point_weight_position = stack(list(list_point_weight_position))
        list_point_weight = stack(list(list_point_weight))

        list_point_fitness = tensor([
            object_hs.get_fitness_base_kpi(
                vector=point_weight,
                kpi_index=list_point_weight_position[index_point_weight, 0, 2]
            )
            for index_point_weight, point_weight in enumerate(list_point_weight)
        ])

        point_pheromone_matrix = tensor([
            [self.pheromone_matrix[depth.item(), row.item(), col.item()]
             for depth, row, col in point_position]
            for point_position in list_point_weight_position
        ])

        total = sum(pow(1/list_point_fitness, self.beta) *
                    pow(sum(point_pheromone_matrix, dim=1), self.alpha))
        prob = pow(1/list_point_fitness, self.beta) * \
            pow(sum(point_pheromone_matrix, dim=1), self.alpha) / total

        _, best_point_index = max(prob, dim=0)

        selected_point = list_reachable_point[best_point_index]
        selected_position_weight = list_point_weight_position[best_point_index]
        selected_weight = list_point_weight[best_point_index]

        return selected_point, selected_position_weight, selected_weight

    def calculate_fitness_base_path(self, ant_path: list, weight_position: Tensor, object_hs: ObjectHarmonySearch, path_solution_candidates: Tensor):
        path_length = list()
        for index, edge in enumerate(ant_path):
            if index + 1 < len(ant_path) - 1:
                path_detail = next(
                    (item for item in path_solution_candidates if item['from'] == edge and item['to'] == ant_path[index + 1]), None)
                harmony_memory = path_detail['harmony_memory'].clone().detach()
                item_position = weight_position[index].clone().detach()

                element = list()
                for depth, row, col in item_position:
                    element.append(
                        tensor(harmony_memory[depth.item(), row.item(), col.item()].clone().detach()))

                path_length.append(object_hs.get_fitness_tensor(
                    tensor(element), ant_path[index + 1]))

        return sum(stack(path_length))
