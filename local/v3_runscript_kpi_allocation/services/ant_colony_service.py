from torch import Tensor
from models import AntColony
from helpers import START_POINT_NAME, FINISH_POINT_NAME
from random import randint
from torch import Tensor
import torch
from models import HarmonySearch
from typing import List
from helpers import Timer
import numpy

timer = Timer()


class AntColonyService(object):
    ant_colony: AntColony

    def __init__(self, ant_colony: AntColony):
        self.ant_colony = ant_colony

    # def get_first_point_duration(self, to_edge: str, listTaskLinkage: list[TaskLinkageRequest], num_row: int) -> Tensor:
    #     task_index = (int(to_edge) - 1) % num_row

    #     kpi_index = next((task_linkage.kpi_metric_id -
    #                      1 for task_linkage in listTaskLinkage if task_linkage.task_id == int(to_edge)), 0)

    #     return torch.sum(self.ant_colony.duration_matrix[task_index, kpi_index])

    def build_ant_weight_base_prob_transit(self, prob_transit: torch.Tensor, num_row: int, num_col: int, harmony_search: HarmonySearch):
        # reshape_tensor = prob_transit.view(hms, -1, num_item)
        max_values, max_indices = torch.max(prob_transit, dim=0)
        max_values = max_values.view(num_row, num_col)
        max_indices = max_indices.view(num_row, num_col)

        positions = torch.nonzero(
            max_values.unsqueeze(0) == prob_transit, as_tuple=False)
        positions = [(pos[0].item(), pos[1].item(), pos[2].item())
                     for pos in positions]

        ant_matrix = torch.zeros(num_row, num_col)
        positions_tensor = torch.tensor(positions)

        # Extract depth, row, col, and item from positions
        depth = positions_tensor[:, 0]
        row = positions_tensor[:, 1]
        col = positions_tensor[:, 2]
        # item = positions_tensor[:, 3]

        # Assign values from harmony_search.harmony_memory to ant_matrix
        ant_matrix[row, col] = harmony_search.harmony_memory[depth, row, col]

        return ant_matrix, positions

    def run_algorithm(self, harmony_search: HarmonySearch, num_row: int, num_col: int, prob_transit: torch.Tensor):
        object_hs = harmony_search.objective_harmony_search

        ant_weight, ant_weight_position = self.build_ant_weight_base_prob_transit(prob_transit=prob_transit,
                                                                                  num_row=num_row, num_col=num_col,
                                                                                  harmony_search=harmony_search)
        fitness = object_hs.get_fitness(ant_weight)

        return ant_weight, ant_weight_position, fitness

    def update_local_pheromone(self, ant_colony: AntColony, prob_transit: torch.Tensor, rho_local: torch.Tensor) -> None:
        default_pheromone = ant_colony.default_pheromone_value

        prob_transit.mul_(1 - rho_local).add_(default_pheromone * rho_local)

    def update_global_pheromone(self, prob_transit: torch.Tensor, rho_global: torch.Tensor, positions: torch.Tensor, fitness: torch.Tensor) -> None:
        for position in positions:
            depth_val, row_val, col_val = position
            prob_transit[depth_val, row_val, col_val] = prob_transit[depth_val, row_val, col_val] * (
                1 - rho_global[row_val, col_val]) + rho_global[row_val, col_val] * (1 / fitness)
