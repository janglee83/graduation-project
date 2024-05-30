from torch import Tensor
from models import HarmonySearch
from random import random
import torch


class HarmonyService(object):
    def __init__(self):
        pass

    def run_algorithm(self, harmony_search: HarmonySearch, lower_upper_matrix: Tensor, num_row: int, num_col: int):
        object_hs = harmony_search.objective_harmony_search

        while True:
            # generate new harmony from each layer
            harmony = torch.zeros(num_row, num_col)

            # for row in range(num_row):
            for col in range(num_col):
                # set bound
                object_hs.set_lower_bound(lower_upper_matrix[col, 0].item())
                object_hs.set_upper_bound(lower_upper_matrix[col, 1].item())

                if random() < object_hs.hmcr:
                    harmony_search.memory_consideration(harmony=harmony, col=col)

                    if random() < object_hs.par:
                        harmony_search.pitch_adjustment(harmony=harmony, col=col)

                else:
                    harmony_search.random_selection(harmony=harmony, col=col)

            fitness = object_hs.get_fitness(harmony=harmony)

            if fitness != float('inf'):
                break

        harmony_search.update_harmony_memory(considered_harmony=harmony, considered_fitness=fitness, num_row=num_row, num_col=num_col)
