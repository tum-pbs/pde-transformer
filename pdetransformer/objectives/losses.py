import torch.nn.functional as F
def loss(type: str, **kwargs):

    if type == "l1":
        f = F.l1_loss
    elif type == "l2":
        f = F.mse_loss
    else:
        raise ValueError("Invalid loss type {type}")

    def _loss(x, pred, noise, *args, split: str = ""):
        l = f(pred, noise)
        return l

    return _loss