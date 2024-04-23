from pydantic import BaseModel
from .object_harmony_search import ObjectHarmonySearch
from typing import List, Optional
from services.truncated_normal_service import TruncatedNormalService
from torch import zeros, Tensor, tensor, max, min
from random import randint, random


class HarmonySearch(BaseModel):
    objective_harmony_search: ObjectHarmonySearch
    harmony_memory: Optional[List]

    def __init__(self, objective_harmony_search: ObjectHarmonySearch, harmony_memory: List = None) -> None:
        super().__init__(objective_harmony_search=objective_harmony_search,
                         harmony_memory=harmony_memory)

    def set_harmony_memory(self, harmony_memory: Tensor) -> None:
        self.harmony_memory = harmony_memory

    def initialize_harmony_memory(self, lower_upper_matrix: Tensor, executive_staff_matrix: Tensor) -> Tensor:
        row_len = len(self.objective_harmony_search.human_score_vector)
        col_len = len(self.objective_harmony_search.kpi_weight)
        truncated_normal = TruncatedNormalService()

        while True:
            harmony = zeros(row_len, col_len)

            for col in range(col_len):
                # build size
                size = (1, (executive_staff_matrix[:, col] == 1).sum().item())

                # define bound object hs
                bound_matrix = lower_upper_matrix[col]
                self.objective_harmony_search.set_lower_bound(bound_matrix[0])
                self.objective_harmony_search.set_upper_bound(bound_matrix[1])
                truncated_normal.set_min_val(bound_matrix[0])
                truncated_normal.set_max_val(bound_matrix[1])

                truncated_weight_matrix = truncated_normal.truncated_normal(
                    size=size)
                # self.objective_harmony_search.get_fitness(truncated_weight_matrix)
                mask = executive_staff_matrix[:, col].nonzero(
                    as_tuple=False).squeeze(1)
                # print(mask.view(-1))
                harmony[mask, col] = truncated_weight_matrix.view(-1)

            fitness = self.objective_harmony_search.get_fitness(
                harmony=harmony)

            if fitness >= 0:
                break

        return harmony

    def memory_consideration(self, harmony: list, row: int, col: int) -> None:
        memory_index = randint(0, self.objective_harmony_search.hms - 1)
        hm = self.harmony_memory.clone().detach()
        harmony[row, col] = hm[memory_index, row, col]

    def memory_consideration_layer(self, layer: Tensor, row: int, col: int) -> None:
        random_layer_index = randint(
            0, self.objective_harmony_search.depth_value - 1)
        random_row_index = randint(0, self.objective_harmony_search.hms - 1)
        layer[row, col] = self.harmony_memory[random_layer_index,
                                              random_row_index, col]

    def random_selection(self, harmony: list, row: int, col: int) -> None:
        harmony[row, col] = self.objective_harmony_search.get_value()

    def random_selection_layer(self, layer: Tensor, row: int, col: int) -> None:
        layer[row, col] = tensor(self.objective_harmony_search.get_value())

    def pitch_adjustment(self, harmony: List, row: int, col: int, lower_bound: float, upper_bound: float) -> None:
        bw = self.objective_harmony_search.bw

        # continuous variable
        adjustment = tensor(random() * bw)

        if random() < 0.5:
            # adjust pitch down
            harmony[row, col] -= adjustment * (harmony[row, col] - lower_bound)
        else:
            # adjust pitch up
            harmony[row, col] += adjustment * (upper_bound - harmony[row, col])

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

    def update_harmony_memory(self, considered_harmony: Tensor, considered_fitness: float) -> None:
        hm_fitness = tensor(list(map(lambda candidate: self.objective_harmony_search.get_fitness(
            candidate), self.harmony_memory)))
        worse_fitness, worse_index = max(hm_fitness, dim=0)
        if considered_fitness < worse_fitness:
            self.harmony_memory[worse_index] = considered_harmony

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
