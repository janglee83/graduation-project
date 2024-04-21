from pydantic import BaseModel
from .object_harmony_search import ObjectHarmonySearch
from typing import List, Optional
from services.truncated_normal_service import TruncatedNormalService
from torch import zeros, Tensor, tensor, any, all, eq, max, min
from random import randint, random, uniform


class HarmonySearch(BaseModel):
    objective_harmony_search: ObjectHarmonySearch
    harmony_memory: Optional[List]

    def __init__(self, objective_harmony_search: ObjectHarmonySearch, harmony_memory: List = None) -> None:
        super().__init__(objective_harmony_search=objective_harmony_search,
                         harmony_memory=harmony_memory)

    def set_harmony_memory(self, harmony_memory: Tensor) -> None:
        self.harmony_memory = harmony_memory

    def initialize_harmony_memory(self) -> Tensor:
        truncated_normal_service = TruncatedNormalService()
        harmonies = zeros(self.objective_harmony_search.depth_value,
                          self.objective_harmony_search.hms, self.objective_harmony_search.number_parameters)
        size = (1, self.objective_harmony_search.number_parameters)

        for depth in range(self.objective_harmony_search.depth_value):
            for row in range(self.objective_harmony_search.hms):
                harmony = list()
                while True:
                    harmony = truncated_normal_service.generate_truncated_normal_samples(
                        size=size,
                        max_val=self.objective_harmony_search.upper_bounds,
                        min_val=self.objective_harmony_search.lower_bounds
                    )
                    fitness = self.objective_harmony_search.get_fitness(
                        harmony.clone().detach())

                    if fitness >= 0:
                        break  # If fitness is non-negative, exit the loop

                harmonies[depth, row] = harmony

        return harmonies

    def memory_consideration(self, harmony: list, col: int, depth: int) -> None:
        memory_index = randint(0, self.objective_harmony_search.hms - 1)
        hm = self.harmony_memory.clone().detach()
        harmony.append(hm[depth, memory_index, col])

    def memory_consideration_layer(self, layer: Tensor, row: int, col: int) -> None:
        random_layer_index = randint(
            0, self.objective_harmony_search.depth_value - 1)
        random_row_index = randint(0, self.objective_harmony_search.hms - 1)
        layer[row, col] = self.harmony_memory[random_layer_index,
                                              random_row_index, col]

    def random_selection(self, harmony: list) -> None:
        harmony.append(tensor(self.objective_harmony_search.get_value()))

    def random_selection_layer(self, layer: Tensor, row: int, col: int) -> None:
        layer[row, col] = tensor(self.objective_harmony_search.get_value())

    def pitch_adjustment(self, harmony: List, col: int) -> None:
        lower_bounds = self.objective_harmony_search.lower_bounds
        upper_bounds = self.objective_harmony_search.upper_bounds
        bw = self.objective_harmony_search.bw

        # continuous variable
        adjustment = tensor(random() * bw)

        if random() < 0.5:
            # adjust pitch down
            harmony[col] -= adjustment * (harmony[col] - lower_bounds)
        else:
            # adjust pitch up
            harmony[col] += adjustment * (upper_bounds - harmony[col])

    def pitch_adjustment_layer(self, layer: Tensor, row: int, col: int) -> None:
        # continuous variable
        if random() < 0.5:
            # adjust pitch down
            layer[row, col] -= tensor(((layer[row, col] - self.objective_harmony_search.lower_bounds)
                                       * random() * self.objective_harmony_search.bw))
        else:
            # adjust pitch up
            layer[row, col] += tensor((self.objective_harmony_search.upper_bounds -
                                       layer[row, col]) * random() * self.objective_harmony_search.bw)

    def update_harmony_memory(self, considered_harmony: Tensor, considered_fitness: float, path_solution_candidate: Tensor, index_path: int, depth: int) -> None:
        layer = self.harmony_memory[depth]
        if not any(all(eq(layer, considered_harmony), dim=1)):
            layer_fitness_value = list()
            for _, col in enumerate(layer):
                layer_fitness_value.append(
                    self.objective_harmony_search.get_fitness(col))

            worst_fitness, worst_index = max(
                tensor(layer_fitness_value), dim=0)
            if considered_fitness < worst_fitness:
                self.harmony_memory[depth, worst_index] = considered_harmony
                path_solution_candidate[index_path]['harmony_memory'][depth,
                                                                      worst_index] = considered_harmony

    def update_harmony_base_layer(self, considered_layer: Tensor, path_solution_candidate: Tensor, index: int) -> None:
        def calculate_layer_fitness_value(layer: Tensor) -> Tensor:
            return tensor([self.objective_harmony_search.get_fitness(row) for row in layer]).view(self.objective_harmony_search.hms, 1)

        fitness_consider_value, fitness_consider_index = min(
            calculate_layer_fitness_value(considered_layer), dim=0)

        for depth in range(self.objective_harmony_search.depth_value):
            layer = self.harmony_memory[depth]

            worse_layer_value, worse_layer_index = max(
                calculate_layer_fitness_value(layer), dim=0)
            if fitness_consider_value < worse_layer_value:
                path_solution_candidate[index]['harmony_memory'][depth, worse_layer_index.item(
                )] = considered_layer[fitness_consider_index][0]
