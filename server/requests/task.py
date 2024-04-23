from pydantic import BaseModel, validator
from typing import List
from .helper import check_string_helper, check_float_helper

class TaskRequest(BaseModel):
    id: str
    name: str
    value: str
    symbol: str

    # @validator('id')
    # def check_string_id(cls, value):
    #     return check_string_helper(value=value, error_message='Id must be string')

    # @validator('name')
    # def check_string_name(cls, value):
    #     return check_string_helper(value=value, error_message='Name must be string')

    # @validator('value')
    # def check_float_value(cls, value):
    #     return check_float_helper(value=value, error_message='Value must be float')

    # @validator('symbol')
    # def check_string_symbol(cls, value):
    #     return check_string_helper(value=value, error_message='Symbol must be float')
