import torch
import torch.nn.functional as F

def triplet_loss(anchor, positive, negative, margin=1.0):
    a_p = torch.sum((anchor - positive) ** 2, dim = -1)
    a_n = torch.sum((anchor - negative) ** 2, dim = -1)
    x = a_p - a_n + margin
    ans = F.relu(x)
    return torch.mean(ans)
    # TODO: mean triplet loss with squared L2 distance
    pass
