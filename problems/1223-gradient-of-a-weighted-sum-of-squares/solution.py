import torch

def grad_wss(w_list, x_list):
    w = torch.tensor(w_list, requires_grad = True)
    x = torch.tensor(x_list)
    L = sum(torch.square(x * w)) * 0.5
    L.backward()
    return w.grad.tolist()
    """Build w (requires_grad) and x from lists, compute
    loss = 0.5 * sum((w * x)**2), backward, return w.grad
    as a list of floats rounded to 4 decimals.
    """
    # TODO
    pass
