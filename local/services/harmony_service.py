from torch import Tensor, zeros, all, sum
from models import HarmonySearch
from random import random, randint


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
            harmony = zeros(row_len, col_len, item_len)

            for row in range(row_len):
                for col in range(col_len):
                    for item in range(item_len):
                        # set bound
                        bound_matrix = lower_upper_matrix[row, col].clone(
                        ).detach()
                        object_hs.set_lower_bound(bound_matrix[0])
                        object_hs.set_upper_bound(bound_matrix[1])

                        # add condition if that employee can not do that kpi
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

            if fitness > 0:
                break

        harmony_search.update_harmony_memory(
            considered_harmony=harmony, considered_fitness=fitness)
