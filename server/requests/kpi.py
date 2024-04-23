from pydantic import BaseModel, field_validator
from typing import List
from .helper import check_string_helper, check_float_helper


class KpiRequest(BaseModel):
    id: str
    name: str
    value: float
    symbol: str
    executive_staff: List[str] | str
    lower_bound: float
    upper_bound: float
    weight: float

    # Task weight matrix where the total weight of all items in a row sums up to 1.
    # If an item's weight is 0, it means that the corresponding task cannot be done by the given kpi_id.
    task_weight: List[float]

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
        if isinstance(v, str):
            v = [v]
        if not all(isinstance(item, str) for item in v):
            raise ValueError(
                "executive_staff must contain only strings or a single string")
        return v

    @field_validator('task_weight')
    def validate_task_weight(cls, v):
        # Check if the list is not empty
        if not v:
            raise ValueError('task_weight cannot be empty')

        # Check if the total weight sums up to 1
        total_weight = sum(v)
        if total_weight != 1:
            raise ValueError('Total weight of task_weight must be 1')

        # Check if all items are non-negative
        if any(item < 0 for item in v):
            raise ValueError('task_weight cannot contain negative values')

        return v

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
