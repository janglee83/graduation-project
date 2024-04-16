from pydantic import BaseModel
from typing import Any
from torch import Tensor, sum, allclose
from helpers import get_vector_rank_number


class ObjectHarmonySearch(BaseModel):
    lower_bounds: float
    upper_bounds: float
    number_parameters: int
    is_maximize: bool
    max_improvisations: int
    hms: int
    hmcr: float
    par: float
    bw: float
    human_score_vector: Any
    kpi_weight: Any

    def __init__(self, number_parameters: int, human_score_vector: Any, kpi_weight: Any, lower_bounds: float = 0.0, upper_bounds: float = 0.2, is_maximize: bool = False, max_improvisations: int = 2, hms: int = 100, hmcr: float = 0.75, par: float = 0.5, bw: float = 0.5):
        super().__init__(lower_bounds=lower_bounds, upper_bounds=upper_bounds, number_parameters=number_parameters, is_maximize=is_maximize,
                         max_improvisations=max_improvisations, hms=hms, hmcr=hmcr, par=par, bw=bw, human_score_vector=human_score_vector, kpi_weight=kpi_weight)

    def get_fitness(self, vector: Tensor) -> float:
        ranked_human_vector = get_vector_rank_number(self.human_score_vector)
        ranked_vector = get_vector_rank_number(vector=vector)

        if not allclose(ranked_human_vector, ranked_vector):
            return float('-inf')

        return sum(self.kpi_weight * self.human_score_vector * vector) - sum(self.kpi_weight)
