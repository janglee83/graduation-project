from torch import Tensor
from models import AntColony
from helpers import START_POINT_NAME, FINISH_POINT_NAME
from random import randint
from torch import Tensor, tensor, zeros
import torch
from models import HarmonySearch, ObjectHarmonySearch
from typing import List
from requests import TaskLinkageRequest
from helpers import Timer
import numpy
from requests import TaskRequest, PRODUCT_FACTOR_TYPE, ENVIRONMENT_FACTOR_TYPE

timer = Timer()


def generate_rho_matrix(num_row: int, num_col: int, listTask: list[TaskRequest], listTaskLinkage: list[TaskLinkageRequest]):
    matrix = torch.zeros(num_row, num_col)
    for task_linkage in listTaskLinkage:
        kpi_index = task_linkage.kpi_metric_id - 1
        task_of_kpi_index = (task_linkage.task_id - 1) % num_row

        env_score_list = list()
        product_score_list = list()

        for task in listTask:
            if task.task_id == task_linkage.task_id:

                for factor in task.affected_factors:
                    tuple_factor = [tuple(factor)]
                    for factor_id, type, score in tuple_factor:
                        if type == PRODUCT_FACTOR_TYPE:
                            product_score_list.append(score)
                        else:
                            env_score_list.append(score)

            env_score_list = [float(score) for score in env_score_list]
            env_score_mean = torch.tensor(env_score_list).mean() / 5

            product_score_list = [float(score) for score in product_score_list]
            product_score_mean = torch.tensor(env_score_list).mean() / 5

            matrix[task_of_kpi_index, kpi_index] = (
                1 - env_score_mean) * (1 - product_score_mean)

    return matrix

class AntColonyService(object):
    ant_colony: AntColony

    def __init__(self, ant_colony: AntColony):
        self.ant_colony = ant_colony

    def get_path_weight_first_path(self, ant_weight: Tensor, harmony_search: HarmonySearch, to_edge: str, num_hms: int, listTaskLinkage: list[TaskLinkageRequest], num_row: int) -> Tensor:
        task_index = (int(to_edge) - 1) % num_row
        rand_hms = randint(0, num_hms - 1)

        kpi_index = next((task_linkage.kpi_metric_id -
                         1 for task_linkage in listTaskLinkage if task_linkage.task_id == int(to_edge)), 0)

        ant_weight[task_index, kpi_index, :] = harmony_search.harmony_memory[rand_hms, task_index, kpi_index, :]

        return torch.tensor((rand_hms, task_index, kpi_index)), torch.sum(self.ant_colony.duration_matrix[task_index, kpi_index])

    def ant_random_first_point(self, hms_len: int, row_len: int, col_len: int, item_len: int):
        def generate_random_int_number(upper_bound: int):
            return randint(0, upper_bound - 1)

        random_hms = generate_random_int_number(hms_len)
        random_row = generate_random_int_number(row_len)
        random_col = generate_random_int_number(col_len)
        random_item = generate_random_int_number(item_len)

        return random_hms, random_row, random_col, random_item

    def find_best_next_point_position(self, harmony_memory, pheromone_matrix):
        probabilities = (harmony_memory ** self.ant_colony.alpha) * (pheromone_matrix **
                                                                     self.ant_colony.beta) * (self.ant_colony.duration_matrix ** (self.ant_colony.beta + 2))
        total = torch.sum(probabilities)
        prob_tensor = probabilities / total
        # Flatten the probability tensor
        flat_probs = prob_tensor.view(-1)

        # Find the index with maximum probability
        max_prob_index = torch.argmax(flat_probs)

        # Convert flattened index to original index
        max_hms = max_prob_index // (
            harmony_memory.shape[1] * harmony_memory.shape[2] * harmony_memory.shape[3])
        max_row = (max_prob_index % (harmony_memory.shape[1] * harmony_memory.shape[2] *
                harmony_memory.shape[3])) // (harmony_memory.shape[2] * harmony_memory.shape[3])
        max_col = ((max_prob_index % (harmony_memory.shape[1] * harmony_memory.shape[2]
                * harmony_memory.shape[3])) // harmony_memory.shape[3]) % harmony_memory.shape[2]
        max_item = (max_prob_index % (
            harmony_memory.shape[1] * harmony_memory.shape[2] * harmony_memory.shape[3])) % harmony_memory.shape[3]

        return max_hms.item(), max_row.item(), max_col.item(), max_item.item()

    def run_algorithm(self, harmony_search: HarmonySearch, num_row: int, num_col: int, num_item: int, listTaskLinkage: list[TaskLinkageRequest], num_task: int):
        timer.start()
        gen_best_path = list()
        object_hs = harmony_search.objective_harmony_search

        # Initialize ant_weight and ant_weight_position outside the loop
        ant_weight = torch.zeros((num_row, num_col, num_item))
        ant_weight_position = []

        for _ in range(self.ant_colony.number_ants):
            ant_weight.zero_()
            ant_weight_position.clear()
            ant_duration = 0.0

            current_ant_path = [START_POINT_NAME, str(
                numpy.random.randint(1, self.ant_colony.number_edge))]

            # Reset ant_weight and ant_weight_position
            position_first_point, current_duration = self.get_path_weight_first_path(
                ant_weight=ant_weight, harmony_search=harmony_search, to_edge=current_ant_path[-1], num_hms=object_hs.hms, listTaskLinkage=listTaskLinkage, num_row=num_row)

            ant_weight_position.append(position_first_point)
            ant_duration += current_duration

            # start find path
            while len(current_ant_path) != self.ant_colony.number_edge + 2:
                reachable_point = self.ant_colony.get_list_available_next_note(
                    current_ant_path[-1], current_ant_path, num_task)

                point_str, position, next_duration = self.ant_colony.get_best_next_point(
                    list_reachable_point=reachable_point, harmony_search=harmony_search, num_row=num_row, listTaskLinkage=listTaskLinkage, antWeight=ant_weight, num_task=num_task)

                if position != []:
                    ant_weight_position.append(position)

                current_ant_path.append(point_str)
                ant_duration += next_duration


            if current_ant_path.count(START_POINT_NAME) == 1 and current_ant_path.count(FINISH_POINT_NAME) == 1:
                fitness_path = object_hs.get_fitness(ant_weight)

                if fitness_path >= 0:
                    payload = {
                        'path_weight': current_ant_path,
                        'weight_position': ant_weight_position,
                        'path_length': fitness_path,
                        'ant_weight': ant_weight.clone(),
                        'duration': ant_duration
                    }

                    gen_best_path.append(payload)
        if gen_best_path:
            return min(gen_best_path, key=lambda x: x['path_length'] if x['path_length'] > 0 else float('inf'))
        return None

    def update_local_pheromone(self, ant_colony: AntColony, listTask: list[TaskRequest], listTaskLinkage: list[TaskLinkageRequest], num_row: int, num_col: int) -> None:
        pheromone_tensor: torch.Tensor = ant_colony.pheromone_matrix.clone().detach()

        pl_matrix = generate_rho_matrix(num_col=num_col, num_row=num_row, listTask=listTask, listTaskLinkage=listTaskLinkage)

        new_pheromone = pheromone_tensor * (1 - pl_matrix) + ant_colony.default_pheromone_value * pl_matrix

        ant_colony.pheromone_matrix = new_pheromone

    def update_global_pheromone(self, ant_colony: AntColony, best_path: dict, listTask: list[TaskRequest], listTaskLinkage: list[TaskLinkageRequest], num_row: int, num_col: int) -> None:
        pheromone_tensor: torch.Tensor = ant_colony.pheromone_matrix.clone().detach()

        pg_matrix = generate_rho_matrix(
            num_col=num_col, num_row=num_row, listTask=listTask, listTaskLinkage=listTaskLinkage)

        new_pheromone = pheromone_tensor * (1 - pg_matrix)

        for kpi_position in best_path['weight_position']:
            kpi_position_tuple = [tuple(kpi_position)]
            for _, row_val, col_val in kpi_position_tuple:
                new_pheromone[:, row_val, col_val] += pg_matrix[row_val, col_val] * (1 / best_path['path_length'])

        ant_colony.pheromone_matrix = new_pheromone
