from pydantic import BaseModel
from typing import Optional, List, Any
from torch import Tensor, pow, sum, max, argmax, unravel_index, nonzero, tensor, cat, stack, zeros
import torch
from helpers import convert_edge_str_to_index
from helpers import FINISH_POINT_NAME
from models import ObjectHarmonySearch


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

    def __init__(self, number_ants: int, number_edge: int, relationship_kpi_matrix: Any, best_ant_path: List = [], best_ant_path_length: float = 0.0, alpha: float = 0.4, beta: float = 0.6, default_pheromone_value: float = 1.0, pl: float = 0.4, pg: float = 0.6) -> None:
        super().__init__(number_ants=number_ants, alpha=alpha, beta=beta, best_ant_path=best_ant_path, best_ant_path_length=best_ant_path_length,
                         number_edge=number_edge, relationship_kpi_matrix=relationship_kpi_matrix, default_pheromone_value=default_pheromone_value, pl=pl, pg=pg)

    def get_weight_base_trans_prob(self, col_layer: Tensor, harmony_pheromone_candidate_value: Tensor, col: int):
        harmony_pheromone_candidate_col = harmony_pheromone_candidate_value[:, :, col].unsqueeze(
            -1).clone().detach()
        total = sum(pow(col_layer, self.beta) *
                    pow(harmony_pheromone_candidate_col, self.alpha))

        pheromone_matrix = (pow(col_layer, self.beta) *
                            pow(harmony_pheromone_candidate_col, self.alpha)) / total
        position = unravel_index(
            argmax(pheromone_matrix), pheromone_matrix.shape)

        position_depth = position[0].item()
        position_row = position[1].item()
        value = col_layer[position_depth, position_row, 0]

        return position_depth, position_row, value

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
            position_depth, position_row, value = self.get_weight_base_trans_prob(
                path_detail['harmony_memory'][:, :, col].unsqueeze(-1), path_detail['harmony_pheromone_candidate_value'], col)
            weight_position.append([position_depth, position_row, col])
            weight_values.append(value)

        return tensor(weight_position), tensor(weight_values)

    def get_best_next_point(self, start_point: str, list_reachable_point: list, path_solution_candidates: Tensor, object_hs: ObjectHarmonySearch, ant_weight_tensor: Tensor, ant_path_index: int):
        if not list_reachable_point:
            return FINISH_POINT_NAME, []

        matrix_weight_vector = list()
        pheromone_matrix = list()
        weight_positions = list()
        for point in list_reachable_point:
            weight_position, weight_value = self.get_path_weight(
                path_solution_candidates, object_hs, start_point, to_edge=point)
            matrix_weight_vector.append(weight_value)

            harmony_pheromone_candidate_value = next((item for item in path_solution_candidates if item['from'] == start_point and item['to'] == point), None)[
                'harmony_pheromone_candidate_value'].clone().detach()

            pheromone_edge_matrix = zeros(object_hs.number_parameters)
            for pos in weight_position:
                depth, row, col = pos
                pheromone_edge_matrix[col.item(
                )] = harmony_pheromone_candidate_value[depth.item(), row.item(), col.item()]

            pheromone_matrix.append(pheromone_edge_matrix.mean())
            weight_positions.append(weight_position)

        pheromone_matrix = stack(pheromone_matrix)
        matrix_weight_vector = stack(matrix_weight_vector)

        fitness_values = tensor(
            [object_hs.get_fitness(item) for item in matrix_weight_vector])
        total = sum(pow(fitness_values, self.beta) *
                    pow(pheromone_matrix, self.alpha))

        prob_matrix = (pow(fitness_values, self.beta) *
                       pow(pheromone_matrix, self.alpha)) / total

        position = unravel_index(argmax(prob_matrix), prob_matrix.shape)

        index_result = position[0].item()

        ant_weight_tensor[ant_path_index] = matrix_weight_vector[index_result]

        return list_reachable_point[index_result], weight_positions[index_result]

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
