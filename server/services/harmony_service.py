from torch import Tensor, zeros
from models import HarmonySearch
from random import random
from services import TruncatedNormalService


class HarmonyService(object):
    def __init__(self):
        pass

    def run_algorithm(self, harmony_search: HarmonySearch, lower_upper_matrix: Tensor):
        truncatedNS = TruncatedNormalService()
        object_hs = harmony_search.objective_harmony_search
        row_len = len(
            harmony_search.objective_harmony_search.human_score_vector)
        col_len = len(harmony_search.objective_harmony_search.kpi_weight)

        while True:
            # generate new harmony from each layer
            harmony = zeros(row_len, col_len)
            for row in range(row_len):
                for col in range(col_len):
                    # set bound
                    bound = lower_upper_matrix[col]
                    object_hs.set_lower_bound(bound[0])
                    object_hs.set_upper_bound(bound[1])

                    # add condition if that employee can not do that kpi
                    if random() < object_hs.hmcr or harmony_search.harmony_memory[1, row, col] == 0:
                        harmony_search.memory_consideration(
                            harmony=harmony, row=row, col=col)

                        if random() < object_hs.par and harmony_search.harmony_memory[1, row, col] != 0:
                            harmony_search.pitch_adjustment(
                                harmony=harmony, row=row, col=col, lower_bound=bound[0], upper_bound=bound[1])

                    else:
                        harmony_search.random_selection(
                            harmony=harmony, row=row, col=col)

            fitness = object_hs.get_fitness(harmony=harmony)
            if truncatedNS.is_truncated_normal(harmony) and fitness >= 0:
                break

        harmony_search.update_harmony_memory(
            considered_harmony=harmony, considered_fitness=fitness)
