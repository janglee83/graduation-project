from pydantic import BaseModel, field_validator
from typing import List
from .helper import check_string_helper, check_float_helper


class TaskRequest(BaseModel):
    id: str
    name: str
    value: float
    symbol: str
    executive_staff: List[str] | str
    lower_bound: float
    upper_bound: float
    weight: float

    @field_validator('id')
    def check_string_id(cls, value):
        return check_string_helper(value=value, error_message='Id must be string')

    @field_validator('name')
    def check_string_name(cls, value):
        return check_string_helper(value=value, error_message='Name must be string')

    @field_validator('value')
    def check_float_value(cls, value):
        return check_float_helper(value=value, error_message='Value must be float')

    @field_validator('symbol')
    def check_string_symbol(cls, value):
        return check_string_helper(value=value, error_message='Symbol must be float')

    @field_validator('lower_bound')
    def check_float_lower_bound(cls, value):
        return check_float_helper(value=value, error_message='Lower bound must be float')

    @field_validator('upper_bound')
    def check_float_upper_bound(cls, value):
        return check_float_helper(value=value, error_message='Upper bound must be float')

    @field_validator("executive_staff")
    def validate_executive_staff(cls, v):
        if not all(isinstance(item, str) for item in v):
            raise ValueError(
                "executive_staff must contain only strings or a single string")
        return v
