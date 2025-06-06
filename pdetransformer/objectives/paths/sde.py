from src.objectives.paths.path_base import PathBase
import torch

class VariancePreservingSDE(PathBase):
    """
    Variance Preserving SDE: alpha(t) = exp(-0.5 * t), sigma(t) = sqrt(1 - alpha(t) ** 2)
    """

    def __init__(self, beta_min = 0.1, beta_max = 20):
        super().__init__()
        self.beta_min = beta_min
        self.beta_max = beta_max

    def T(self, t):
        return t * self.beta_min + 0.5 * (t ** 2) * (self.beta_max-self.beta_min)

    def alpha(self, t):
        return torch.exp(-0.5 * self.T(1-t))

    def sigma(self, t):
        return torch.sqrt(1 - self.alpha(t) ** 2)

    def grad_alpha(self, t):
        # TODO
        pass

    def grad_sigma(self, t):
        # TODO
        pass

class VarianceExplodingSDE(PathBase):
    """
    Variance Exploding SDE: alpha(t) = 1, sigma(t) = sigma_max * ((sigma_min / sigma_max) ** t)
    """

    def __init__(self, sigma_max: float, sigma_min: float):
        super().__init__()
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min

    def alpha(self, t):
        return torch.ones_like(t, dtype=t.dtype)

    def sigma(self, t):
        return self.sigma_max * torch.power((self.sigma_min / self.sigma_max), t)

    def grad_alpha(self, t):
        # TODO
        pass

    def grad_sigma(self, t):
        # TODO
        pass