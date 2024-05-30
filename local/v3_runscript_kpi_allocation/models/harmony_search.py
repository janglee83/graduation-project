from pydantic import BaseModel
from .object_harmony_search import ObjectHarmonySearch
from typing import List, Optional
from services.truncated_normal_service import TruncatedNormalService
from torch import Tensor, tensor, max
import torch
from requests import TaskAssignRequest


class HarmonySearch(BaseModel):
    objective_harmony_search: ObjectHarmonySearch
    harmony_memory: Optional[List]

    def __init__(self, objective_harmony_search: ObjectHarmonySearch, harmony_memory: List = None) -> None:
        super().__init__(objective_harmony_search=objective_harmony_search,
                         harmony_memory=harmony_memory)

    def set_harmony_memory(self, harmony_memory: Tensor) -> None:
        self.harmony_memory = harmony_memory

    def initialize_harmony_memory(self, lower_upper_tensor: Tensor, num_row: int, num_col: int):
        truncated_normal = TruncatedNormalService()

        task_size_tensor = torch.zeros(num_row, num_col, dtype=torch.long)

        lower_bounds = lower_upper_tensor[:, 0]
        upper_bounds = lower_upper_tensor[:, 1]

        # Initialize harmony memory tensor
        while True:
            harmony = torch.zeros(num_row, num_col)

            for col in range(num_col):
                size_len = num_row

                # Define bounds
                lower_bound = lower_bounds[col].item()
                upper_bound = upper_bounds[col].item()
                self.objective_harmony_search.set_lower_bound(lower_bound)
                self.objective_harmony_search.set_upper_bound(upper_bound)
                truncated_normal.set_min_val(lower_bound)
                truncated_normal.set_max_val(upper_bound)

                # Generate truncated normal distribution
                kpi_unit_weight = truncated_normal.generate_truncated_normal_with_sum(size=size_len)
                harmony[:, col] = kpi_unit_weight

            fitness = self.objective_harmony_search.get_fitness(harmony=harmony)
            if fitness != float('inf'):
                break

        return harmony

    def memory_consideration(self, harmony: torch.Tensor, col: int) -> None:
        memory_index = torch.randint(
            0, self.objective_harmony_search.hms, (1,))
        hm: torch.Tensor = torch.Tensor(self.harmony_memory)
        harmony[:, col] = hm[memory_index, :, col]

    def random_selection(self, harmony: Tensor, col: int) -> None:
        harmony[:, col] = self.objective_harmony_search.get_value()

    def pitch_adjustment(self, harmony: Tensor, col: int) -> None:
        bw = self.objective_harmony_search.bw
        lower_bound = self.objective_harmony_search.lower_bound
        upper_bound = self.objective_harmony_search.upper_bound

        # continuous variable
        adjustment = (torch.rand(1) * bw)[0]

        if torch.rand(1) < 0.5:
            # adjust pitch down
            harmony[:, col] -= adjustment * (harmony[:, col] - lower_bound)
        else:
            # adjust pitch up
            harmony[:, col] += adjustment * (upper_bound - harmony[:, col])

    def update_harmony_memory(self, considered_harmony: Tensor, considered_fitness: float, num_row: int, num_col: int) -> None:
        # # consider to use?
        # for col in range(num_col):
        #     current_kpi_fitness = self.objective_harmony_search.get_fitness_base_kpi(
        #         considered_harmony[:, col, :])
        #
        #     current_kpi_hm_fitness = list()
        #     for hs in range(self.objective_harmony_search.hms):
        #         current_kpi_hm_fitness.append(self.objective_harmony_search.get_fitness_base_kpi(
        #             self.harmony_memory[hs, :, col, :]))
        #
        #     current_kpi_hm_fitness = torch.tensor(current_kpi_hm_fitness)
        #     worse_fitness_kpi, worse_index_hs = torch.max(
        #         current_kpi_hm_fitness, dim=0)
        #     if current_kpi_fitness < worse_fitness_kpi:
        #         self.harmony_memory[worse_index_hs, :,
        #         col] = considered_harmony[:, col]

        hm_fitness = tensor(list(map(lambda candidate: self.objective_harmony_search.get_fitness(
            candidate), self.harmony_memory)))
        worse_fitness, worse_index = max(hm_fitness, dim=0)
        if considered_fitness < worse_fitness:
            self.harmony_memory[worse_index] = considered_harmony
