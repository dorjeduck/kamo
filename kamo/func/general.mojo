from algorithm import vectorize

from kamo import MN,MM,MV,SD,SD2

alias ACF = fn (SD, Bool=False) -> SD
alias ACF_MV = fn (MV, Bool=False) -> MV

fn identity(x: SD, grad: Bool = False) -> SD:
    if not grad:
        return x
    else:
        return 1.0

fn silu(x: SD, grad: Bool = False) -> SD:
    if not grad:
        return x / (1.0 + MN.exp(-x))
    else:
        return (1.0 + MN.exp(-x) + x * MN.exp(-x)) / MN.pow(1.0 + MN.exp(-x))

fn relu(x: SD, grad: Bool = False) -> SD:
    if not grad:
        return 0 if x < 0 else x
    else:
        return 0 if x < 0 else 1.0

fn tanh_act(x: SD, grad: Bool = False) -> SD:
    if not grad:
        return MN.tanh(x)  # math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
    return 1 - MN.tanh(x) ** 2  # tanh_act(x, grad=False) ** 2

fn sigmoid_act(x: SD, grad: Bool = False) -> SD:
    if not grad:
        return 1 / (1 + MN.exp(-x))
    return sigmoid_act(x) * (1 - sigmoid_act(x))

fn identity_mv(x: MV, grad: Bool = False) -> MV:
    if not grad:
        return x
    else:
        return MV(x.size, 1.0)

fn silu_mv(x: MV, grad: Bool = False) -> MV:
    if not grad:
        return x / (1.0 + MN.exp(-x))
    else:
        var sigmoid_x = MV(x.size,1.0) / (SD(1.0) + MN.exp(-x))
        return sigmoid_x * (1.0 + x * (1.0 - sigmoid_x))

fn relu_mv(x: MV, grad: Bool = False) -> MV:
    var res = MV(x.size)
    if not grad:

        @parameter
        fn _relu[width: Int](iv: Int):
            var mask = x.load[width=width](iv) > 0
            res.store[width=width](iv, mask.select(x.load[width=width](iv), 0.0))

        vectorize[_relu, simd_width](size=x.size)

    else:

        @parameter
        fn _relu_grad[width: Int](iv: Int):
            var mask = x.load[width=width](iv) > 0
            res.store[width=width](iv, mask.select(SD(1.0), 0.0))

        vectorize[_relu_grad, simd_width](size = x.size)

    return res


fn tanh_act_mv(x: MV, grad: Bool = False) -> MV:
    if not grad:
        return MN.tanh(x)  # math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
    return 1 - MN.tanh(x) ** 2  # tanh_act(x, grad=False) ** 2


fn sigmoid_act_mv(x: MV, grad: Bool = False) -> MV:
    if not grad:
        return MN.pow(1 + MN.exp(-x),-1)
    return sigmoid_act_mv(x) * (1 - sigmoid_act_mv(x))
