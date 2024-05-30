def check_string_helper(value: str, error_message: str) -> str:
    if not isinstance(value, str):
        raise ValueError(error_message)

    return value


def check_float_helper(value: float, error_message: str) -> float:
    if not isinstance(value, float):
        raise ValueError(error_message)

    return value


def check_int_helper(value: int, error_message: str) -> float:
    if not isinstance(value, int):
        raise ValueError(error_message)

    return value
