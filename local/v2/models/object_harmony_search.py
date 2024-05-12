from pydantic import BaseModel
from typing import Any, Optional
from torch import Tensor
import torch
from random import uniform
from os import getenv
from services.truncated_normal_service import TruncatedNormalService


class ObjectHarmonySearch(BaseModel):
    lower_bound: Optional[float]
    upper_bound: Optional[float]
    number_parameters: int
    max_improvisations: int
    hms: int
    hmcr: float
    par: float
    bw: float
    task_kpi_weight_vector: Any
    lower_upper_matrix: Any

    def __init__(self, number_parameters: int, task_kpi_weight_vector: Any, lower_upper_matrix: Any, max_improvisations: int = 2, hms: int = 2, hmcr: float = 0.75, par: float = 0.5, bw: float = 0.5, lower_bound: float = None, upper_bound: float = None):
        max_improvisations = 1750
        hms = 20
        hmcr = 0.75
        par = 0.5
        bw = 0.5

        super().__init__(number_parameters=number_parameters, max_improvisations=max_improvisations, hms=hms, hmcr=hmcr, par=par, bw=bw , lower_bound=lower_bound, upper_bound=upper_bound, task_kpi_weight_vector=task_kpi_weight_vector, lower_upper_matrix=lower_upper_matrix)

    def update_hmcr_par_base_gen(self, gen: int) -> None:
        # self.hmcr = 0.9 + 0.2
        frac_value = torch.tensor((gen - 1) / (self.max_improvisations - 1))
        sqrt_value = torch.sqrt(frac_value * (1 - frac_value))

        # self.hmcr = torch.tensor(0.9 + 0.2 * sqrt_value)
        # self.par = torch.tensor(0.85 + 0.3 * sqrt_value)
        # print(self.hmcr, self.par)
        # print(frac_value.item())

    def get_fitness(self, harmony: Tensor) -> float:
        penalty = 0
        task_weight = harmony.sum(dim=2)
        if (task_weight < 1).any():
            penalty += float('+inf')

        # # print(torch.sum(torch.max(torch.zeros_like(task_weight), 1 - task_weight)))
        fitness = task_weight.sum() + penalty
        # print(fitness)

        return fitness

    def get_fitness_base_kpi(self, vector: Tensor, kpi_index: int) -> float:
        # print(vector)
        task_weight = vector.sum(dim=1)
        penalty = 0
        if (task_weight < 1).any():
            penalty += float('+inf')

        fitness = task_weight.sum() + penalty
        return fitness
        # if (vector.sum(dim=1) >= 1).all:
        #     return torch.sum(vector.sum(dim=1) - 1)
        # return float('inf')

    def get_value(self) -> float:
        t_normal = TruncatedNormalService(min_val=self.lower_bound, max_val=self.upper_bound)
        size = (1)
        return t_normal.generate_truncated_normal_with_sum(size=size)

    def set_lower_bound(self, lower_bound: float) -> None:
        self.lower_bound = lower_bound

    def set_upper_bound(self, upper_bound: float) -> None:
        self.upper_bound = upper_bound
