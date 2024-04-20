from torch import Tensor, tensor, zeros
from models import HarmonySearch
from random import random, randint
from services import TruncatedNormalService


class HarmonyService(object):
    def __init__(self):
        pass

    def run_algorithm(self, path_solution_candidates: Tensor, harmony_search: HarmonySearch):
        truncatedNS = TruncatedNormalService()

        for index, path in enumerate(path_solution_candidates):
            harmony_search.set_harmony_memory(path['harmony_memory'])

            object_hs = harmony_search.objective_harmony_search

            # generate new harmony from each layer
            for depth in range(object_hs.depth_value):
                while True:
                    harmony: list = list()
                    for col in range(object_hs.number_parameters):
                        if random() < object_hs.hmcr:
                            harmony_search.memory_consideration(
                                harmony, col, depth)

                            if random() < object_hs.par:
                                harmony_search.pitch_adjustment(
                                    harmony=harmony, col=col)
                        else:
                            harmony_search.random_selection(harmony=harmony)

                    tensor_harmony = tensor(harmony)
                    fitness = object_hs.get_fitness(tensor_harmony)

                    if truncatedNS.is_truncated_normal(tensor_harmony) and fitness >= 0:
                        break

                harmony_search.update_harmony_memory(
                    tensor_harmony, fitness, path_solution_candidates, index, depth)

            # generate new layer from layers
            new_layer = zeros(object_hs.hms, object_hs.number_parameters)

            for row in range(object_hs.hms):
                while True:
                    harmony: list = list()
                    for col in range(0, object_hs.number_parameters):
                        if random() < object_hs.hmcr:
                            harmony_search.memory_consideration(
                                harmony, col, depth)

                            if random() < object_hs.par:
                                harmony_search.pitch_adjustment(
                                    harmony=harmony, col=col)
                        else:
                            harmony_search.random_selection(harmony=harmony)

                        tensor_harmony = tensor(harmony)
                        fitness = object_hs.get_fitness(tensor_harmony)

                    if truncatedNS.is_truncated_normal(tensor_harmony) and fitness >= 0:
                        break

                new_layer[row] = tensor(harmony)

            harmony_search.update_harmony_base_layer(
                new_layer, path_solution_candidates, index)
