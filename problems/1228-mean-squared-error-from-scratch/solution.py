import torch

def mse(pred, target):
    ans = (pred - target) ** 2
    return torch.mean(ans).item()
    """
    Compute mean squared error between pred and target.

    Args:
        pred (torch.Tensor): Predicted values.
        target (torch.Tensor): Ground-truth values (same shape as pred).

    Returns:
        float: Mean of squared differences.
    """
    # TODO
    pass
