import torch


def get_vector_rank_number(vector: torch.Tensor) -> torch.Tensor:
    sorted_vector, indices = torch.sort(vector)
    ranked_vector = torch.zeros_like(vector)

    current_rank = 1
    for i in range(len(vector)):
        # So sánh với một ngưỡng rất nhỏ
        if i > 0 and torch.abs(sorted_vector[i] - sorted_vector[i - 1]) < 1e-6:
            ranked_vector[indices[i]] = ranked_vector[indices[i - 1]]
        else:
            ranked_vector[indices[i]] = current_rank
            current_rank += 1

    return ranked_vector
