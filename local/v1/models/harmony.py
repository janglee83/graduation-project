from pydantic import BaseModel
from typing import List
from torch import tensor


class Harmony(BaseModel):
    list_weight: List[float]
    fitness_value: float

    def __init__(self, listWeight: List[float], fitnessValue: float):
        super().__init__(list_weight=listWeight, fitness_value=fitnessValue)

    def to_tensor(self):
        return tensor(self.list_weight), tensor(self.fitness_value)
