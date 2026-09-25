import math
import torch

def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    position = torch.arange(seq_len)
    position = position.reshape(seq_len, 1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000) / d_model))
    temp = position * div_term
    ans = torch.zeros(seq_len, d_model)
    ans[:, 0::2] = torch.sin(temp)
    ans[:, 1::2] = torch.cos(temp)
    return ans
    # TODO: return a (seq_len, d_model) tensor of sinusoidal positional encodings
    pass
