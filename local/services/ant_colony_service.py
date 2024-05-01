from torch import Tensor
from models import AntColony
from helpers import START_POINT_NAME, FINISH_POINT_NAME
from random import randint
from torch import Tensor, tensor, zeros
import torch
from models import HarmonySearch, ObjectHarmonySearch
from requests import EnvironmentEffect, EquipmentEffect, HumanEffect, ProductEffect
from typing import List


class AntColonyService(object):
    ant_colony: AntColony

    def __init__(self, ant_colony: AntColony):
        self.ant_colony = ant_colony

    def get_path_weight_first_path(self, ant_weight: Tensor, harmony_search: HarmonySearch, to_edge: str) -> Tensor:
        kpi_index = int(to_edge) - 1

        weight_position = list()
        for row in range(3):
            for item in range(len(harmony_search.objective_harmony_search.human_score_vector)):
                position_harmony_memory, weight = self.ant_colony.get_weight_item_base_rand_hms(
                    harmony_search=harmony_search, row=row, col=kpi_index, item=item)

                weight_position.append(
                    [position_harmony_memory, row, kpi_index, item])
                ant_weight[row, kpi_index, item] = weight

        return tensor(weight_position)

    def ant_random_first_point(self, hms_len: int, row_len: int, col_len: int, item_len: int):
        def generate_random_int_number(upper_bound: int):
            return randint(0, upper_bound - 1)

        random_hms = generate_random_int_number(hms_len)
        random_row = generate_random_int_number(row_len)
        random_col = generate_random_int_number(col_len)
        random_item = generate_random_int_number(item_len)

        return random_hms, random_row, random_col, random_item

    def find_best_next_point_position(self, harmony_memory: torch.Tensor, pheromone_matrix: torch.Tensor):
        total = torch.sum(torch.pow(harmony_memory, self.ant_colony.alpha) * torch.pow(pheromone_matrix, self.ant_colony.beta))
        prob_tensor = torch.pow(harmony_memory, self.ant_colony.alpha) * torch.pow(pheromone_matrix, self.ant_colony.beta) / total

        max_prob_index = torch.argmax(prob_tensor)
        max_hms, max_row, max_col, max_item = torch.unravel_index(max_prob_index, prob_tensor.shape)
        return max_hms.item(), max_row.item(), max_col.item(), max_item.item()

    def run_algorithm(self, harmony_search: HarmonySearch):
        gen_best_path = list()
        object_hs = harmony_search.objective_harmony_search
        row_len = 3
        col_len = len(object_hs.kpi_weight_vector)
        item_len = len(object_hs.human_score_vector)

        for _ in range(self.ant_colony.number_ants):
            # generate local storage
            ant_weight: Tensor = zeros(row_len, col_len, item_len)
            ant_weight_position = list()

            # clone harmony search memory and its pheromone
            harmony_memory: torch.Tensor = harmony_search.harmony_memory.clone().detach()
            pheromone_matrix: torch.Tensor = self.ant_colony.pheromone_matrix.clone().detach()
            pheromone_matrix[(harmony_memory == 0.0)] = 0.0

            # put ant in random point of harmon search result
            random_hms, random_row, random_col, random_item = self.ant_random_first_point(hms_len=object_hs.hms, row_len=row_len, col_len=col_len, item_len=item_len)

            # put result into local storage
            ant_weight[random_row, random_col, random_item] = harmony_memory[random_hms, random_row, random_col, random_item]
            ant_weight_position.append((random_row, random_row, random_col, random_item))

            # delete result which had been found from harmony memory
            harmony_memory[:, random_row, random_col, random_item] = 0.0
            pheromone_matrix[:, random_row, random_col, random_item] = 0.0

            # check condition if all item in pheromone is equal to 0
            while (pheromone_matrix != 0).any():
                max_hms, max_row, max_col, max_item = self.find_best_next_point_position(harmony_memory=harmony_memory, pheromone_matrix=pheromone_matrix)

                # put result into local storage
                ant_weight[max_row, max_col, max_item] = harmony_memory[max_hms, max_row, max_col, max_item]
                ant_weight_position.append((max_hms, max_row, max_col, max_item))

                # delete result which had been found from harmony memory
                harmony_memory[:, max_row, max_col, max_item] = 0.0
                pheromone_matrix[:, max_row, max_col, max_item] = 0.0

            fitness_path = object_hs.get_fitness(ant_weight)

            if fitness_path > 0:
                payload = {
                    'weight_position': torch.tensor(ant_weight_position),
                    'path_length': fitness_path,
                    'ant_weight': ant_weight
                }

            gen_best_path.append(payload)



        # for _ in range(self.ant_colony.number_ants):
        #     current_ant_path: list = [START_POINT_NAME, str(
        #         randint(1, self.ant_colony.number_edge))]

        #     ant_weight: Tensor = zeros(row_len, col_len, item_len)
        #     ant_weight_position = list()

        #     position_first_path = self.get_path_weight_first_path(
        #         ant_weight=ant_weight, harmony_search=harmony_search, to_edge=current_ant_path[-1])

        #     ant_weight_position.append(position_first_path)

        #     # start find path
        #     while len(current_ant_path) != self.ant_colony.number_edge + 2:
        #         reachable_point = self.ant_colony.get_list_available_next_note(
        #             current_ant_path[-1], current_ant_path)
        #         point, weight_point_position, weight_point = self.ant_colony.get_best_next_point(
        #             list_reachable_point=reachable_point, harmony_search=harmony_search)
        #         current_ant_path.append(point)
        #         if ant_weight_position != [] and weight_point != []:
        #             ant_weight_position.append(weight_point_position)
        #             kpi_index = weight_point_position[0, 2]
        #             ant_weight[:, kpi_index] = weight_point.clone().detach()

        #     if current_ant_path.count(START_POINT_NAME) == 1 and current_ant_path.count(FINISH_POINT_NAME) == 1:
        #         fitness_path = object_hs.get_fitness(ant_weight)

        #         if fitness_path > 0:
        #             payload = {
        #                 'path_weight': current_ant_path,
        #                 'weight_position': ant_weight_position,
        #                 'path_length': fitness_path,
        #                 'ant_weight': ant_weight
        #             }

        #             gen_best_path.append(payload)

        if gen_best_path:
            return min(gen_best_path, key=lambda x: x['path_length'] if x['path_length'] > 0 else float('inf'))
        return None

    def update_local_pheromone(self, ant_colony: AntColony, listEnvironmentsEffectScore: List[EnvironmentEffect], listEquipmentEffectScore: List[EquipmentEffect], object_harmony_search: ObjectHarmonySearch) -> None:
        pheromone_tensor: torch.Tensor = ant_colony.pheromone_matrix.clone().detach()
        pl_matrix = torch.zeros_like(pheromone_tensor)

        hms_len = object_harmony_search.hms
        row_len = 3 #Task
        col_len = ant_colony.number_edge

        def find_score(list_to_find: List[EnvironmentEffect | EquipmentEffect], kpi_id: str, task_id: str):
            list_score = torch.tensor([task.score
                  for item in list_to_find
                  for kpi in item.list_kpi
                  if kpi.kpi_id == kpi_id
                  for task in kpi.tasks
                  if task.task_id == task_id])

            return list_score.mean() / 5

        for hms in range(hms_len):
            for row in range(row_len):
                for col in range(col_len):
                    env_score = find_score(listEnvironmentsEffectScore, str(col + 1), str(row + 1))
                    eq_score = find_score(listEquipmentEffectScore, str(col + 1), str(row + 1))

                    pl = (torch.tensor(1.00000001) - env_score) * (torch.tensor(1.00000001) - eq_score)

                    pl_matrix[hms, row, col, :] = pl

        pheromone_tensor = (1 - pl_matrix) * pheromone_tensor + ant_colony.default_pheromone_value * pl_matrix

        ant_colony.pheromone_matrix = pheromone_tensor

    def update_global_pheromone(self, ant_colony: AntColony, best_path: dict, listHumanEffectScore: List[HumanEffect], listProductEffectScore: List[ProductEffect], object_harmony_search: ObjectHarmonySearch) -> None:
        pheromone_tensor = ant_colony.pheromone_matrix.clone().detach()
        pg_matrix = torch.zeros_like(pheromone_tensor)

        hms_len = object_harmony_search.hms
        col_len = ant_colony.number_edge

        def find_score(list_to_find: List[EnvironmentEffect | EquipmentEffect], kpi_id: str):
            list_score = torch.tensor([task.score
                  for item in list_to_find
                  for kpi in item.list_kpi
                  if kpi.kpi_id == kpi_id
                  for task in kpi.tasks])

            return list_score.mean() / 5

        for hms in range(hms_len):
            for col in range(col_len):
                human_score = find_score(listHumanEffectScore, str(col + 1))
                product_score = find_score(listProductEffectScore, str(col + 1))

                pg = (torch.tensor(1.00000001) - human_score) * (torch.tensor(1.00000001) - product_score)
                pg_matrix[hms, :, col, :] = pg

        pheromone_tensor = (1 - pg_matrix) * pheromone_tensor

        for kpi_position in best_path['weight_position']:
            depth_val = kpi_position[0].item()
            row_val = kpi_position[1].item()
            col_val = kpi_position[2].item()
            item_val = kpi_position[3].item()

            pheromone_tensor[depth_val, row_val, col_val, item_val] += pg_matrix[depth_val, row_val, col_val, item_val] * (1 / best_path['path_length'])

        ant_colony.pheromone_matrix = pheromone_tensor
