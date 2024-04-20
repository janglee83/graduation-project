import torch
from gluonts.torch.distributions.truncated_normal import TruncatedNormal


class TruncatedNormalService(object):
    def __init__(self) -> None:
        pass

    def truncated_normal(self, size, std=0.01, min_val=None, max_val=None):
        if min_val is None:
            min_val = 0
        if max_val is None:
            max_val = 1

        mean = min_val + torch.rand(1).item() * (max_val - min_val)
        tensor = torch.normal(mean, std, size=size)
        # Ensure all values are greater than 0
        tensor = torch.clamp(tensor, min=1e-9)
        if min_val is not None or max_val is not None:
            if min_val is None:
                min_val = float('-inf')
            if max_val is None:
                max_val = float('inf')
            tensor = torch.clamp(tensor, min=min_val, max=max_val)
        return tensor

    def generate_truncated_normal_samples(self, size, std=0.01, min_val=0, max_val=0.2):
        samples = self.truncated_normal(
            size, std=std, min_val=min_val, max_val=max_val)
        return samples[0]

    def is_truncated_normal(self, vector: torch.Tensor, lower=0, upper=1, tolerance=1e-5):
        mu = torch.mean(vector.clone().detach()).requires_grad_(True)  # mean
        sigma = torch.tensor(0.1).requires_grad_(True)  # standard deviation

        # Create a truncated normal distribution object
        trunc_norm = TruncatedNormal(mu, sigma, lower, upper)

        # Calculate the cumulative probabilities of lower and upper bounds
        cdf_lower = trunc_norm.cdf(lower)
        cdf_upper = trunc_norm.cdf(upper)

        # Calculate the probability within the specified range
        truncated_prob = cdf_upper - cdf_lower

        # Check if the probability is close to 1
        return abs(truncated_prob - 1.0) < tolerance
