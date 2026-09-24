import torch

def grad_of_quadratic(x_value: float) -> float:
    input = torch.tensor(x_value, requires_grad = True)
    y = input ** 2 + 3 * input + 2
    y.backward()
    return input.grad.item()
    # TODO: build a tracked leaf for x, compute f(x), run backprop, return df/dx as a float
    pass
