from abc import ABC, abstractmethod
import torch

class PathBase(ABC):
    """
    Base class for paths. All consider the transformation x_t = alpha(t) * x_1 + sigma(t) * epsilon, where epsilon is
    standard Gaussian noise.
    """
    def __init__(self):
        pass

    def snr(self, t):
        """
        Compute the signal-to-noise ratio at time t.
        :param t:
        :return:
        """
        return 2 * (torch.log(self.alpha(t)) - torch.log(self.sigma(t)))

    def grad_snr(self, t):
        pass
        # TODO jax to torch grad
        # return vmap(grad(self.snr))(t)

    @abstractmethod
    def alpha(self, t):
        pass

    @abstractmethod
    def sigma(self, t):
        pass

    def grad_sigma(self, t):
        pass
        # TODO jax to torch grad
        # return vmap(grad(self.sigma))(t)

    def grad_alpha(self, t):
        pass
        # TODO jax to torch grad
        # return vmap(grad(self.alpha))(t)