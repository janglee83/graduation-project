import torch
import torch.nn as nn
from gluonts.torch.distributions.truncated_normal import TruncatedNormal
from pydantic import BaseModel


class TruncatedNormalService(BaseModel):
    min_val: float
    max_val: float

    def __init__(self, min_val: float = 0, max_val: float = 1) -> None:
        super().__init__(min_val=min_val, max_val=max_val)

    def set_min_val(self, min_val: float) -> None:
        self.min_val = min_val

    def set_max_val(self, max_val: float) -> None:
        self.max_val = max_val

    def truncated_normal(self, size, std=0.01):
        mean = self.min_val + \
            torch.rand(1).item() * (self.max_val - self.min_val)
        tensor = torch.normal(mean, std, size=size)
        # Ensure all values are greater than 0
        tensor = torch.clamp(tensor, min=1e-9)

        tensor = torch.clamp(tensor, min=self.min_val, max=self.max_val)

        # print(tensor)
        tensor = tensor[0]
        print(tensor.sum() < 1)

        # # Ensure the sum of tensor is greater than 1
        # while tensor.sum() < 1:
        #     tensor += torch.normal(mean, std, size=size)

        # return tensor[0]

    def generate_truncated_normal_with_sum(self, size: torch.Tensor, min_sum: int = 1, std_dev=0.01):
        shape = (1, size)
        mean_tensor = (self.min_val + self.max_val) / 2

        result = torch.empty(shape)
        nn.init.trunc_normal_(result, mean=mean_tensor,
                              std=std_dev, a=self.min_val, b=self.max_val)

        return result[0]

    def is_truncated_normal(self, harmony: torch.Tensor, tolerance=1e-5):
        sigma = torch.tensor(0.1).requires_grad_(True)  # standard deviation

        for col in range(harmony.size(1)):
            tensor_check = harmony[:, col][harmony[:, col] != 0]
            mu = torch.mean(tensor_check.clone().detach()
                            ).requires_grad_(True)  # mean

            # Create a truncated normal distribution object
            trunc_norm = TruncatedNormal(mu, sigma, self.min_val, self.max_val)

            # Calculate the cumulative probabilities of lower and upper bounds
            cdf_lower = trunc_norm.cdf(self.min_val)
            cdf_upper = trunc_norm.cdf(self.max_val)

            # Calculate the probability within the specified range
            truncated_prob = cdf_upper - cdf_lower

            if abs(truncated_prob - 1.0) > tolerance:
                return False

        return True
