from pydantic import BaseModel, field_validator
from .helper import check_string_helper, check_float_helper


class EmployeeRequest(BaseModel):
    id: str
    score: float
    task_completion_rate: float

    @field_validator('id')
    def check_string_id(cls, value):
        return check_string_helper(value=value, error_message='Id must be string')

    @field_validator('score')
    def check_float_point(cls, value):
        return check_float_helper(
            value=value, error_message='Point must be a float value')

    @field_validator('task_completion_rate')
    def check_float_task_completion_rate(cls, value):
        return check_float_helper(
            value=value, error_message='Task Completion rate must be a float value')
