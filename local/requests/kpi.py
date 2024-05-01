from pydantic import BaseModel, field_validator
from typing import List
from .helper import check_string_helper, check_float_helper
from .task import TaskRequest


class KpiRequest(BaseModel):
    id: str
    name: str
    value: float
    symbol: str
    weight: float
    tasks: List[TaskRequest]

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

    @field_validator('weight')
    def check_float_weight(cls, value):
        return check_float_helper(value=value, error_message='Upper bound must be float')


class KpiConditionRequest(BaseModel):
    id: str
    post_condition: List[str]

    def post_condition_validator(cls, post_condition):
        if isinstance(post_condition, list):
            for item in post_condition:
                cls.check_string_post_condition(
                    value=item, error_message='All items in post_condition must be strings')
        else:
            cls.check_string_post_condition(
                value=post_condition, error_message='All items in post_condition must be strings')

    @staticmethod
    def check_string_post_condition(value, error_message):
        if not isinstance(value, str):
            raise ValueError(error_message)

    @field_validator('id')
    def check_string_id(cls, value):
        return check_string_helper(value=value, error_message='Id must be string')
