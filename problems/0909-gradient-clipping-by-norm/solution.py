import torch

def clip_grad_norm(parameters, max_norm: float) -> float:
    g = [i.grad for i in parameters if i.grad != None]
    g_2 = torch.norm(torch.stack([i.detach().norm() for i in g]))
    if g_2 > max_norm:
        for i in g:
            i.mul_(max_norm / g_2)
    return g_2.item()
    # TODO: compute total grad norm, scale in-place if it exceeds max_norm, return original norm
    pass
