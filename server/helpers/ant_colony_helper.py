from .matrix import START_POINT_NAME, FINISH_POINT_NAME


def convert_edge_str_to_index(edge: str, num_edges: int) -> int:
    if edge == START_POINT_NAME:
        return 0

    if edge == FINISH_POINT_NAME:
        return num_edges + 1

    return int(edge)
