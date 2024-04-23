from torch import Tensor
from models import AntColony
from helpers import START_POINT_NAME, FINISH_POINT_NAME
from random import randint
from torch import Tensor, tensor, zeros
from models import HarmonySearch


class AntColonyService(object):
    ant_colony: AntColony

    def __init__(self, ant_colony: AntColony):
        self.ant_colony = ant_colony

    def get_path_weight_first_path(self, ant_weight: Tensor, harmony_search: HarmonySearch, to_edge: str) -> Tensor:
        kpi_index = int(to_edge) - 1

        weight_position = list()
        for row in range(len(harmony_search.objective_harmony_search.human_score_vector)):
            position_harmony_memory, weight = self.ant_colony.get_weight_base_rand_hms(
                harmony_search=harmony_search, row=row, col=kpi_index)

            # print(position_harmony_memory, weight)
            weight_position.append([position_harmony_memory, row, kpi_index])
            ant_weight[row, kpi_index] = weight

        return tensor(weight_position)

    def run_algorithm(self, harmony_search: HarmonySearch):
        gen_best_path = list()
        object_hs = harmony_search.objective_harmony_search

        for _ in range(self.ant_colony.number_ants):
            current_ant_path: list = [START_POINT_NAME, str(
                randint(1, self.ant_colony.number_edge))]

            ant_weight: Tensor = zeros(
                len(object_hs.human_score_vector), len(object_hs.kpi_weight))
            ant_weight_position = list()

            position_first_path = self.get_path_weight_first_path(
                ant_weight=ant_weight, harmony_search=harmony_search, to_edge=current_ant_path[-1])

            ant_weight_position.append(position_first_path)

            # start find path
            while len(current_ant_path) != self.ant_colony.number_edge + 2:
                reachable_point = self.ant_colony.get_list_available_next_note(
                    current_ant_path[-1], current_ant_path)
                point, weight_point_position, weight_point = self.ant_colony.get_best_next_point(
                    start_point=current_ant_path[-1], list_reachable_point=reachable_point, harmony_search=harmony_search, ant_weight=ant_weight)
                current_ant_path.append(point)
                if ant_weight_position != [] and weight_point != []:
                    ant_weight_position.append(weight_point_position)
                    kpi_index = weight_point_position[0, 2]
                    ant_weight[:, kpi_index] = weight_point.clone().detach()

            if current_ant_path.count(START_POINT_NAME) == 1 and current_ant_path.count(FINISH_POINT_NAME) == 1:
                fitness_path = object_hs.get_fitness(ant_weight)

                if fitness_path >= 0:
                    payload = {
                        'path_weight': current_ant_path,
                        'weight_position': ant_weight_position,
                        'path_length': fitness_path,
                        'ant_weight': ant_weight
                    }

                    gen_best_path.append(payload)

        if gen_best_path:
            return min(gen_best_path, key=lambda x: x['path_length'] if x['path_length'] > 0 else float('inf'))
        return None

    def update_local_pheromone(self, ant_colony: AntColony) -> None:
        pheromone_tensor = ant_colony.pheromone_matrix.clone().detach()

        # demonstrate pl matrix
        pl_matrix = tensor([ant_colony.pl, ant_colony.pl,
                            ant_colony.pl, ant_colony.pl, ant_colony.pl])

        new_pheromone_tensor = (1 - pl_matrix) * pheromone_tensor + \
            pl_matrix * ant_colony.default_pheromone_value
        ant_colony.pheromone_matrix = new_pheromone_tensor

    def update_global_pheromone(self, ant_colony: AntColony, best_path: dict) -> None:
        pheromone_tensor = ant_colony.pheromone_matrix.clone().detach()

        new_pheromone_tensor = (1 - ant_colony.pg) * pheromone_tensor

        for kpi_position in best_path['weight_position']:
            for depth, row, col in kpi_position:
                depth_val = depth.item()
                row_val = row.item()
                col_val = col.item()
                new_pheromone_tensor[depth_val, row_val, col_val] = 1 / best_path['path_length'] * \
                    new_pheromone_tensor[depth_val,
                                         row_val, col_val] * self.ant_colony.pg

        ant_colony.pheromone_matrix = new_pheromone_tensor
