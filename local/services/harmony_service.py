from torch import Tensor
from models import HarmonySearch
from random import random, randint
import torch


class HarmonyService(object):
    def __init__(self):
        pass

    def run_algorithm(self, harmony_search: HarmonySearch, lower_upper_matrix: Tensor):
        object_hs = harmony_search.objective_harmony_search
        row_len = 3  # TODO: Change
        col_len = len(
            harmony_search.objective_harmony_search.kpi_weight_vector)
        item_len = len(
            harmony_search.objective_harmony_search.human_score_vector)

        while True:
            # generate new harmony from each layer
            harmony = torch.zeros(row_len, col_len, item_len)

            for row in range(row_len):
                for col in range(col_len):
                        # set bound
                        object_hs.set_lower_bound(
                            lower_upper_matrix[row, col, 0])
                        object_hs.set_upper_bound(
                            lower_upper_matrix[row, col, 1])

                        for item in range(item_len):
                            if random() < object_hs.hmcr:
                                harmony_search.memory_consideration(
                                    harmony=harmony, row=row, col=col, item=item)

                                if random() < object_hs.par:
                                    harmony_search.pitch_adjustment(
                                        harmony=harmony, row=row, col=col, item=item)

                            else:
                                harmony_search.random_selection(
                                    harmony=harmony, row=row, col=col, item=item)

            fitness = object_hs.get_fitness(harmony=harmony)

            if fitness != float('inf'):
                break

        harmony_search.update_harmony_memory(
            considered_harmony=harmony, considered_fitness=fitness)
