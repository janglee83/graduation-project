from pydantic import BaseModel
from typing import Any, Optional
from torch import Tensor, sum, tensor, mean
import torch
from random import uniform
from os import getenv


class ObjectHarmonySearch(BaseModel):
    lower_bound: Optional[float]
    upper_bound: Optional[float]
    number_parameters: int
    depth_value: int
    is_maximize: bool
    max_improvisations: int
    hms: int
    hmcr: float
    par: float
    bw: float
    human_score_vector: Any
    kpi_weight_vector: Any
    task_kpi_weight_vector: Any
    lower_upper_matrix: Any

    def __init__(self, number_parameters: int, human_score_vector: Any, kpi_weight_vector: Any, task_kpi_weight_vector: Any, lower_upper_matrix: Any, is_maximize: bool = False, max_improvisations: int = 2, hms: int = 2, hmcr: float = 0.75, par: float = 0.5, bw: float = 0.5, depth_value: int = 2, lower_bound: float = None, upper_bound: float = None):
        max_improvisations = 5000
        hms = 30
        hmcr = float(getenv('HARMONY_SEARCH_HMCR'))
        par = float(getenv('HARMONY_SEARCH_PAR'))
        bw = float(getenv('HARMONY_SEARCH_BW'))
        depth_value = float(getenv('HARMONY_SEARCH_HM_DEPTH'))

        super().__init__(number_parameters=number_parameters, is_maximize=is_maximize, max_improvisations=max_improvisations, hms=hms, hmcr=hmcr, par=par, bw=bw, human_score_vector=human_score_vector, kpi_weight_vector=kpi_weight_vector,
                         depth_value=depth_value, lower_bound=lower_bound, upper_bound=upper_bound, task_kpi_weight_vector=task_kpi_weight_vector, lower_upper_matrix=lower_upper_matrix)

    def get_fitness(self, harmony: Tensor) -> float:
        # print(torch.sum(torch.sum(harmony.sum(dim=2) *
        #       self.task_kpi_weight_vector, dim=0) * self.kpi_weight_vector) - 1)
        # print(torch.mean(harmony.sum(dim=2)) - 1)
        # print(torch.sum(harmony.sum(dim=2) * self.task_kpi_weight_vector, dim=0).mean() - 1)
        # print((torch.sum(harmony.sum(dim=2) *
        #       self.task_kpi_weight_vector, dim=0) * self.kpi_weight_vector).mean())
        # return torch.sum(torch.sum(harmony.sum(dim=2) * self.task_kpi_weight_vector, dim=0) * self.kpi_weight_vector) - 1 + (torch.mean(harmony.sum(dim=2)) - 1) + (torch.sum(harmony.sum(dim=2) * self.task_kpi_weight_vector, dim=0).mean() - 1)
        # if (harmony.sum(dim=2) >= 1).all().item() is True:
            # if (torch.sum(harmony.sum(dim=2) * self.task_kpi_weight_vector, dim=0) >= 1).all().item() is True:
            #         if torch.sum(torch.sum(harmony.sum(dim=2) * self.task_kpi_weight_vector, dim=0) * self.kpi_weight_vector >= 1).all().item() is True:
            return torch.sum(torch.sum(harmony.sum(dim=2) * self.task_kpi_weight_vector, dim=0) * self.kpi_weight_vector) - 1

        # return float('inf')

    def get_fitness_base_kpi(self, vector: Tensor, kpi_index: int) -> float:
        # print(vector.sum(dim=1).mean() - 1)
        # return sum(vector.sum(dim=1) * self.task_kpi_weight_vector[:, kpi_index]) - 1 + (vector.sum(dim=1).mean() - 1)
        # if (vector.sum(dim=1) >= 1).all().item() is True:
            #     if sum(vector.sum(dim=1) * self.task_kpi_weight_vector[:, kpi_index]) >= 1:
            return sum(vector.sum(dim=1) * self.task_kpi_weight_vector[:, kpi_index]) - 1

        # return float('inf')

    def get_value(self) -> float:
        return uniform(self.lower_bound, self.upper_bound)

    def set_lower_bound(self, lower_bound: float) -> None:
        self.lower_bound = lower_bound

    def set_upper_bound(self, upper_bound: float) -> None:
        self.upper_bound = upper_bound
