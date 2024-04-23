from pydantic import BaseModel
from typing import Any, Optional
from torch import Tensor, sum
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
    kpi_weight: Any

    def __init__(self, number_parameters: int, human_score_vector: Any, kpi_weight: Any, is_maximize: bool = False, max_improvisations: int = 2, hms: int = 2, hmcr: float = 0.75, par: float = 0.5, bw: float = 0.5, depth_value: int = 2, lower_bound: float = None, upper_bound: float = None):
        max_improvisations = 5000
        hms = 50
        hmcr = float(getenv('HARMONY_SEARCH_HMCR'))
        par = float(getenv('HARMONY_SEARCH_PAR'))
        bw = float(getenv('HARMONY_SEARCH_BW'))
        depth_value = float(getenv('HARMONY_SEARCH_HM_DEPTH'))

        super().__init__(number_parameters=number_parameters, is_maximize=is_maximize, max_improvisations=max_improvisations, hms=hms, hmcr=hmcr, par=par,
                         bw=bw, human_score_vector=human_score_vector, kpi_weight=kpi_weight, depth_value=depth_value, lower_bound=lower_bound, upper_bound=upper_bound)

    def get_fitness(self, harmony: Tensor) -> float:
        return sum(self.kpi_weight * harmony) - sum(self.kpi_weight)

    def get_fitness_tensor(self, tensor: Tensor, to_kpi: str) -> Tensor:
        return sum(self.kpi_weight[int(to_kpi) - 1] *
                   tensor) - self.kpi_weight[int(to_kpi) - 1]

    def get_fitness_base_kpi(self, vector: Tensor, kpi_index: int) -> float:
        return sum(self.kpi_weight[kpi_index] * vector) - sum(self.kpi_weight[kpi_index])

    def get_value(self) -> float:
        return uniform(self.lower_bound, self.upper_bound)

    def set_lower_bound(self, lower_bound: float) -> None:
        self.lower_bound = lower_bound

    def set_upper_bound(self, upper_bound: float) -> None:
        self.upper_bound = upper_bound
