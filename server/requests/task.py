from pydantic import BaseModel, field_validator
from typing import List
from .helper import check_string_helper, check_float_helper


class TaskRequest(BaseModel):
    id: str
    name: str
    value: float
    symbol: str

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
