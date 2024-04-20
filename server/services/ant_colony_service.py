from torch import Tensor
from models import AntColony
from helpers import START_POINT_NAME, FINISH_POINT_NAME
from random import randint
from torch import Tensor, tensor, zeros
from models import ObjectHarmonySearch


class AntColonyService(object):
    ant_colony: AntColony

    def __init__(self, ant_colony: AntColony):
        self.ant_colony = ant_colony

    def get_path_weight_first_path(self, ant_weight_tensor: Tensor, path_solution_candidates: Tensor, object_hs: ObjectHarmonySearch, from_edge: str, to_edge: str, ant_path_index: int) -> Tensor:
        path_detail = next(
            (item for item in path_solution_candidates if item['from'] == from_edge and item['to'] == to_edge), None)

        weight_position = list()
        for col in range(object_hs.number_parameters):
            position_depth, position_row, value = self.ant_colony.get_weight_base_trans_prob(
                path_detail['harmony_memory'][:, :, col].unsqueeze(-1), path_detail['harmony_pheromone_candidate_value'], col)
            weight_position.append([position_depth, position_row, col])
            ant_weight_tensor[ant_path_index, col] = value

        return tensor(weight_position)

    def run_algorithm(self, path_solution_candidates: Tensor, number_params: int, object_hs: ObjectHarmonySearch):
        gen_best_path = list()

        for ant in range(self.ant_colony.number_ants):
            current_ant_path: list = [START_POINT_NAME, str(
                randint(1, self.ant_colony.number_edge))]

            ant_weight_tensor: Tensor = zeros(
                self.ant_colony.number_edge, number_params)
            ant_weight_position = list()

            position_first_path = self.get_path_weight_first_path(ant_weight_tensor, path_solution_candidates, object_hs,
                                                                  current_ant_path[-2], current_ant_path[-1], current_ant_path.index(current_ant_path[-1]))
            ant_weight_position.append(position_first_path)

            # start find path
            while len(current_ant_path) != self.ant_colony.number_edge + 2:
                reachable_point = self.ant_colony.get_list_available_next_note(
                    current_ant_path[-1], current_ant_path)

                selected_edge, position = self.ant_colony.get_best_next_point(
                    current_ant_path[-1], reachable_point, path_solution_candidates, object_hs, ant_weight_tensor, current_ant_path.index(current_ant_path[-1]) - 1)

                current_ant_path.append(selected_edge)
                if position != []:
                    ant_weight_position.append(position)

            if current_ant_path.count(START_POINT_NAME) == 1 and current_ant_path.count(FINISH_POINT_NAME) == 1:
                path_length = self.ant_colony.calculate_fitness_base_path(
                    current_ant_path, ant_weight_position, object_hs, path_solution_candidates)

                if path_length > 0:
                    payload = {
                        'path': current_ant_path,
                        'weight_position': ant_weight_position,
                        'path_length': path_length,
                    }
                    gen_best_path.append(payload)

        if gen_best_path:
            return min(gen_best_path, key=lambda x: x['path_length'] if x['path_length'] > 0 else float('inf'))
        return None

    def update_local_pheromone(self, path_solution_candidates: Tensor) -> None:
        for index, path in enumerate(path_solution_candidates):
            harmony_pheromone_path = path['harmony_pheromone_candidate_value'].clone(
            ).detach()
            harmony_pheromone_path = (
                1 - self.ant_colony.pl) * harmony_pheromone_path + self.ant_colony.pl * harmony_pheromone_path
            path_solution_candidates[index]['harmony_pheromone_candidate_value'] = harmony_pheromone_path

    def update_global_pheromone(self, path_solution_candidates: Tensor, best_path_position: Tensor, best_weight: Tensor, best_path: list) -> None:
        for index, path in enumerate(path_solution_candidates):
            harmony_pheromone_path = path['harmony_pheromone_candidate_value'].clone(
            ).detach()
            path_solution_candidates[index]['harmony_pheromone_candidate_value'] = (
                1 - self.ant_colony.pg) * harmony_pheromone_path

        for index, edge in enumerate(best_path):
            if index + 1 < len(best_path) - 1:
                path_detail = None
                path_index = None
                for index_path, item in enumerate(path_solution_candidates):
                    if item['from'] == edge and item['to'] == best_path[index + 1]:
                        path_detail = item
                        path_index = index_path
                        break

                position = best_path_position[index].clone().detach()
                harmony_pheromone_path = path_detail['harmony_pheromone_candidate_value'].clone(
                ).detach()
                for depth, row, col in position:
                    depth_val = depth.item()
                    row_val = row.item()
                    col_val = col.item()
                    harmony_pheromone_path[depth_val, row_val, col_val] = 1 / best_weight * \
                        harmony_pheromone_path[depth_val,
                                               row_val, col_val] * self.ant_colony.pg
                    path_solution_candidates[path_index]['harmony_pheromone_candidate_value'] = harmony_pheromone_path

        # print(path_solution_candidates)
