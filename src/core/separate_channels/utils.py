import torch

def accumulate_embeddings_and_values(*args):

    embedding_list = []
    value_list = []

    for dict_ in args:
        for key, value in dict_.items():

            embedding_list.append(value['embedding'])
            value_list.append(value['value'])

    return embedding_list, value_list

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

def to_patch_representation(images: torch.Tensor, patch_size: int = 16):
    if len(images.shape) == 3:
        images = images.unsqueeze(1)
    batch_size, channels, height, width = images.shape
    kh, kw = patch_size, patch_size  # kernel size
    dh, dw = patch_size, patch_size  # stride
    patches = images.unfold(2, kh, dh).unfold(3, kw, dw)
    patches = patches.contiguous().view(batch_size, -1, channels * kh * kw)

    return patches


def from_patch_representation(patches: torch.Tensor, height: int, width: int, patch_size: int = 16):
    batch_size = patches.shape[0]
    unfold_shape = (batch_size, -1, height // patch_size, width // patch_size, patch_size, patch_size)
    patches_orig = patches.view(unfold_shape)
    output_h = unfold_shape[2] * unfold_shape[4]
    output_w = unfold_shape[3] * unfold_shape[5]
    patches_orig = patches_orig.permute(0, 1, 2, 4, 3, 5).contiguous()
    patches_orig = patches_orig.view(batch_size, -1, output_h, output_w)

    return patches_orig