from pydantic import BaseModel, validator
from typing import List
from .helper import check_string_helper, check_float_helper


class EffectEnvPoint(BaseModel):
    kpi_id: str
    point: float

    @validator('kpi_id')
    def check_string_id(cls, value):
        return check_string_helper(
            value=value, error_message='Kpi Id must be a string')

    @validator('point')
    def check_float_point(cls, value):
        return check_float_helper(
            value=value, error_message='Point must be a float value')


class Environments(BaseModel):
    id: str
    name: str
    effect_point: List[EffectEnvPoint]

    @validator('id')
    def check_string_id(cls, value):
        return check_string_helper(value=value, error_message='Id must be string')

    @validator('name')
    def check_string_name(cls, value):
        return check_string_helper(value=value, error_message='Name must be string')


class KpiOutput(BaseModel):
    id: str
    name: str
    effect_point: List[EffectEnvPoint]

    @validator('id')
    def check_string_id(cls, value):
        return check_string_helper(value=value, error_message='Id must be string')

    @validator('name')
    def check_string_name(cls, value):
        return check_string_helper(value=value, error_message='Name must be string')


class Equipment(BaseModel):
    id: str
    name: str
    effect_point: List[EffectEnvPoint]

    @validator('id')
    def check_string_id(cls, value):
        return check_string_helper(value=value, error_message='Id must be string')

    @validator('name')
    def check_string_name(cls, value):
        return check_string_helper(value=value, error_message='Name must be string')
