from pydantic import BaseModel
from typing import Any
from torch import Tensor, sum, allclose
from helpers import get_vector_rank_number
from random import uniform


class ObjectHarmonySearch(BaseModel):
    lower_bounds: float
    upper_bounds: float
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

    def __init__(self, number_parameters: int, human_score_vector: Any, kpi_weight: Any, lower_bounds: float = 0.0, upper_bounds: float = 0.3, is_maximize: bool = False, max_improvisations: int = 2000, hms: int = 20, hmcr: float = 0.75, par: float = 0.5, bw: float = 0.08, depth_value: int = 10):
        super().__init__(lower_bounds=lower_bounds, upper_bounds=upper_bounds, number_parameters=number_parameters, is_maximize=is_maximize,
                         max_improvisations=max_improvisations, hms=hms, hmcr=hmcr, par=par, bw=bw, human_score_vector=human_score_vector, kpi_weight=kpi_weight, depth_value=depth_value)

    def get_fitness(self, vector: Tensor) -> float:
        return sum(self.kpi_weight * sum(vector)) - sum(self.kpi_weight)

    def get_fitness_tensor(self, tensor: Tensor, to_kpi: str) -> Tensor:
        return sum(self.kpi_weight[int(to_kpi) - 1] *
                   tensor) - self.kpi_weight[int(to_kpi) - 1]

    def get_value(self) -> float:
        return uniform(self.lower_bounds, self.upper_bounds)
