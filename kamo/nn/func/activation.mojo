import math
from mokan import dtype

alias SD = Scalar[dtype]

fn relu(x: SD, grad: Bool = False) -> SD:
    if not grad:
        return 0 if x < 0 else x
    else:
        return 0 if x < 0 else 1.0

fn tanh_act(x: SD, grad: Bool = False) -> SD:
    if not grad:
        return math.tanh(x)  # math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
    return 1 - math.tanh(x) ** 2  # tanh_act(x, grad=False) ** 2


fn sigmoid_act(x: SD, grad: Bool = False) -> SD:
    if not grad:
        return 1 / (1 + math.exp(-x))
    return sigmoid_act(x) * (1 - sigmoid_act(x))
