from pydantic import BaseModel
from .object_harmony_search import ObjectHarmonySearch
from typing import List, Optional
from truncated_normal_service import TruncatedNormalService
from torch import Tensor, tensor, max
import torch


class HarmonySearch(BaseModel):
    objective_harmony_search: ObjectHarmonySearch
    harmony_memory: Optional[List]

    def __init__(self, objective_harmony_search: ObjectHarmonySearch, harmony_memory: List = None) -> None:
        super().__init__(objective_harmony_search=objective_harmony_search,
                         harmony_memory=harmony_memory)

    def set_harmony_memory(self, harmony_memory: Tensor) -> None:
        self.harmony_memory = harmony_memory

    def initialize_harmony_memory(self, lower_upper_matrix: Tensor, num_row: int, num_col: int, num_item: int) -> Tensor:
        truncated_normal = TruncatedNormalService()

        # Initialize harmony memory tensor
        while True:
            harmony = torch.zeros(num_row, num_col, num_item)

            for row in range(num_row):
                for col in range(num_col):
                    #         size = torch.sum(
                    #             executive_task_staff_matrix[row, col] > 0).item()

                    # Define bounds
                    lower_bound = lower_upper_matrix[row, col, 0].item()
                    upper_bound = lower_upper_matrix[row, col, 1].item()
                    self.objective_harmony_search.set_lower_bound(lower_bound)
                    self.objective_harmony_search.set_upper_bound(upper_bound)
                    truncated_normal.set_min_val(lower_bound)
                    truncated_normal.set_max_val(upper_bound)

                    # Generate truncated normal distribution
                    task_weight = truncated_normal.generate_truncated_normal_with_sum(
                        size=num_item)

            #         # Apply task weights to the executive_task_staff_matrix
            #         non_zero_indices = executive_task_staff_matrix[row, col].nonzero(
            #             as_tuple=True)
            #         tensor_task_weight = torch.zeros_like(
            #             executive_task_staff_matrix[row, col])
            #         tensor_task_weight[non_zero_indices] = task_weight[:size]

                    harmony[row, col] = task_weight

            fitness = self.objective_harmony_search.get_fitness(
                harmony=harmony)
            if fitness != float('inf'):
                break

        return harmony
        # return '123'

    def memory_consideration(self, harmony: list, row: int, col: int, item: int) -> None:
        memory_index = torch.randint(
            0, self.objective_harmony_search.hms, (1,))
        hm = self.harmony_memory
        harmony[row, col, item] = hm[memory_index, row, col, item]

    def random_selection(self, harmony: Tensor, row: int, col: int, item: int) -> None:
        if self.harmony_memory[0, row, col, item] != 0:
            harmony[row, col, item] = self.objective_harmony_search.get_value()

    def pitch_adjustment(self, harmony: Tensor, row: int, col: int, item: int) -> None:
        if self.harmony_memory[0, row, col, item] != 0:
            bw = self.objective_harmony_search.bw
            lower_bound = self.objective_harmony_search.lower_bound
            upper_bound = self.objective_harmony_search.upper_bound

            # continuous variable
            adjustment = (torch.rand(1) * bw)[0]

            if torch.rand(1) < 0.5:
                # adjust pitch down
                harmony[row, col, item] -= adjustment * \
                    (harmony[row, col, item] - lower_bound)
            else:
                # adjust pitch up
                harmony[row, col, item] += adjustment * \
                    (upper_bound - harmony[row, col, item])

    def update_harmony_memory(self, considered_harmony: Tensor, considered_fitness: float, num_row: int, num_col: int, num_item: int) -> None:
        # consider to use?
        for col in range(num_col):
            current_kpi_fitness = self.objective_harmony_search.get_fitness_base_kpi(
                considered_harmony[:, col, :])

            current_kpi_hm_fitness = list()
            for hs in range(self.objective_harmony_search.hms):
                current_kpi_hm_fitness.append(self.objective_harmony_search.get_fitness_base_kpi(
                    self.harmony_memory[hs, :, col, :]))

            current_kpi_hm_fitness = torch.tensor(current_kpi_hm_fitness)
            worse_fitness_kpi, worse_index_hs = torch.max(
                current_kpi_hm_fitness, dim=0)
            if current_kpi_fitness < worse_fitness_kpi:
                self.harmony_memory[worse_index_hs, :,
                                    col] = considered_harmony[:, col]

        hm_fitness = tensor(list(map(lambda candidate: self.objective_harmony_search.get_fitness(
            candidate), self.harmony_memory)))
        worse_fitness, worse_index = max(hm_fitness, dim=0)
        if considered_fitness < worse_fitness:
            self.harmony_memory[worse_index] = considered_harmony

        return 0
