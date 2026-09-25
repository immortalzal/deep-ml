import torch

def dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    if training == False or p == 0.0:
        return x
    m = (torch.rand_like(x) >= p).float() / (1 - p)
    return m * x
    # TODO: implement inverted dropout
    pass
