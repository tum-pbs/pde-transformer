from src.objectives.paths.path_base import PathBase
import torch
class ConditionalOptimalTransport(PathBase):
    """
    Conditional Optimal Transport: alpha(t) = t, sigma(t) = 1 - (1-sigma_min) * t
    """

    def __init__(self, sigma_min=1e-4):

        super().__init__()
        self.sigma_min = sigma_min

    def alpha(self, t):
        return t

    def sigma(self, t):
        return 1 - (1 - self.sigma_min) * t

    def grad_alpha(self, t):
        return torch.zeros_like(t)

    def grad_sigma(self, t):
        return - (1 - self.sigma_min) * torch.ones_like(t)
