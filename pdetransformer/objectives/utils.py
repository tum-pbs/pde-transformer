import torch

def get_weighting_function(weighting: str):
    """
    Get weighting function for the loss
    :param weighting: type
    :return:
    """

    def identity(t, w):
        return w

    def constant(t, w):
        return torch.ones_like(t)

    if weighting == "id":
        return identity
    elif weighting == "constant":
        return constant
    else:
        raise ValueError(f"Weighting function {weighting} not implemented")

def sample_time(device, num: int = 1, t0: float = 0.0,
                t1: float = 1.0, eps=1e-5) -> torch.Tensor:
    """
    Sample time points between t0 and t1 based on batch size for variance reduction
    :param device: device
    :param num: batch size
    :param t0: minimum time
    :param t1: maximum time
    :param eps: make sure to not sample exactly at t0 or t1
    :return:
    """

    t0 += eps
    t1 -= eps
    times = torch.linspace(t0, t1, num+1, device=device)[:num]
    uniform = torch.rand((num,), device=device)
    uniform = uniform * ((t1 - t0) / num)
    time_samples = times + uniform

    return time_samples