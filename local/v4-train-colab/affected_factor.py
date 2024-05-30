from pydantic import BaseModel, field_validator
from .request_helper import check_string_helper, check_int_helper

PRODUCT_FACTOR_TYPE: str = 'Product'
ENVIRONMENT_FACTOR_TYPE: str = 'Environment'

class Factor(BaseModel):
    id: int
    description: str

    def __init__(self, id: int, description: str):
        super().__init__(id=id, description=description)

    @field_validator('id')
    def check_int_id(cls, value):
        return check_int_helper(value=value, error_message='Id must be int')

    @field_validator('description')
    def check_string_description(cls, value):
        return check_string_helper(value=value, error_message='Description must be string')


class ProductFactor(Factor):
    def __init__(self, id: int, description: str):
        super().__init__(id=id, description=description)


class EnvironmentFactor(Factor):
    def __init__(self, id: int, description: str):
        super().__init__(id=id, description=description)
