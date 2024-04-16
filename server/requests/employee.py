from pydantic import BaseModel, validator
from .helper import check_string_helper, check_float_helper


class Employees(BaseModel):
    id: str
    name: str
    point: float

    @validator('id')
    def check_string_id(cls, value):
        return check_string_helper(value=value, error_message='Id must be string')

    @validator('name')
    def check_string_name(cls, value):
        return check_string_helper(
            value=value, error_message='Name must be string')

    @validator('point')
    def check_float_point(cls, value):
        return check_float_helper(
            value=value, error_message='Point must be a float value')
