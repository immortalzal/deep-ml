import torch

def layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim = -1, keepdim = True)
    var = x.var(dim = -1, keepdim = True, unbiased = False)
    return gamma * (x - mean) / torch.sqrt(var + eps) + beta
    # TODO: normalize over the last dim, then affine-transform with gamma and beta
    pass
