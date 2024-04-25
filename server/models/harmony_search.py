from pydantic import BaseModel
from .object_harmony_search import ObjectHarmonySearch
from typing import List, Optional
from services.truncated_normal_service import TruncatedNormalService
from torch import zeros, Tensor, tensor, max, min, zeros_like
from random import randint, random


class HarmonySearch(BaseModel):
    objective_harmony_search: ObjectHarmonySearch
    harmony_memory: Optional[List]

    def __init__(self, objective_harmony_search: ObjectHarmonySearch, harmony_memory: List = None) -> None:
        super().__init__(objective_harmony_search=objective_harmony_search,
                         harmony_memory=harmony_memory)

    def set_harmony_memory(self, harmony_memory: Tensor) -> None:
        self.harmony_memory = harmony_memory

    def initialize_harmony_memory(self, lower_upper_matrix: Tensor, executive_task_staff_matrix: Tensor) -> Tensor:
        row_len = 3  # TODO: change 3 into max task
        col_len = len(self.objective_harmony_search.kpi_weight_vector)
        item_len = len(self.objective_harmony_search.human_score_vector)
        truncated_normal = TruncatedNormalService()

        while True:
            harmony = zeros(row_len, col_len, item_len)

            for row in range(row_len):
                for col in range(col_len):
                    size = (executive_task_staff_matrix[row, col].clone(
                    ).detach() > 0).sum().item()

                    # define bound object
                    bound_matrix = lower_upper_matrix[row, col].clone(
                    ).detach()
                    self.objective_harmony_search.set_lower_bound(
                        bound_matrix[0])
                    self.objective_harmony_search.set_upper_bound(
                        bound_matrix[1])
                    truncated_normal.set_min_val(bound_matrix[0])
                    truncated_normal.set_max_val(bound_matrix[1])

                    task_weight = truncated_normal.generate_truncated_normal_with_sum(
                        size=size)
                    tensor_task_weight = zeros_like(
                        executive_task_staff_matrix[row, col])
                    tensor_task_weight[executive_task_staff_matrix[row, col] != 0] = task_weight[:sum(
                        executive_task_staff_matrix[row, col] != 0)]

                    harmony[row, col] = tensor_task_weight.clone().detach()

            fitness = self.objective_harmony_search.get_fitness(
                harmony=harmony)
            if fitness >= 0 and fitness != float('inf'):
                break

        return harmony

    def memory_consideration(self, harmony: list, row: int, col: int, item: int) -> None:
        memory_index = randint(0, self.objective_harmony_search.hms - 1)
        hm = self.harmony_memory.clone().detach()
        harmony[row, col, item] = hm[memory_index, row, col, item]

    def memory_consideration_layer(self, layer: Tensor, row: int, col: int) -> None:
        random_layer_index = randint(
            0, self.objective_harmony_search.depth_value - 1)
        random_row_index = randint(0, self.objective_harmony_search.hms - 1)
        layer[row, col] = self.harmony_memory[random_layer_index,
                                              random_row_index, col]

    def random_selection(self, harmony: Tensor, row: int, col: int, item: int) -> None:
        if self.harmony_memory[0, row, col, item] != 0:
            harmony[row, col, item] = self.objective_harmony_search.get_value()

    def random_selection_layer(self, layer: Tensor, row: int, col: int) -> None:
        layer[row, col] = tensor(self.objective_harmony_search.get_value())

    def pitch_adjustment(self, harmony: Tensor, row: int, col: int, item: int) -> None:
        if self.harmony_memory[0, row, col, item] != 0:
            bw = self.objective_harmony_search.bw
            lower_bound = self.objective_harmony_search.lower_bound
            upper_bound = self.objective_harmony_search.upper_bound

            # continuous variable
            adjustment = tensor(random() * bw)

            if random() < 0.5:
                # adjust pitch down
                harmony[row, col, item] -= adjustment * \
                    (harmony[row, col, item] - lower_bound)
            else:
                # adjust pitch up
                harmony[row, col, item] += adjustment * \
                    (upper_bound - harmony[row, col, item])

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
        print('123')

        def calculate_layer_fitness_value(layer: Tensor) -> Tensor:
            return tensor([self.objective_harmony_search.get_fitness(row) for row in layer]).view(self.objective_harmony_search.hms, 1)

        fitness_consider_value, fitness_consider_index = min(
            calculate_layer_fitness_value(considered_layer), dim=0)

        for depth in range(self.objective_harmony_search.hms):
            layer = self.harmony_memory[depth]

            worse_layer_value, worse_layer_index = max(
                calculate_layer_fitness_value(layer), dim=0)
            print(max(
                calculate_layer_fitness_value(layer), dim=0))
            if fitness_consider_value < worse_layer_value:
                path_solution_candidate[index]['harmony_memory'][depth, worse_layer_index.item(
                )] = considered_layer[fitness_consider_index][0]
