from kamo import dtype, simd_width
from kamo.libs.monum import MoVector

alias SD = Scalar[dtype]
alias MV = MoVector[dtype, simd_width]


trait EdgeFunc:
    fn __init__(inout self, x_bounds: List[SD], n_func: Int):
        pass

    fn __call__(self, x: MV, grad: Bool = False) -> MV:
        pass

    fn __copyinit__(inout self, existing: Self):
        pass

    fn __moveinit__(inout self, owned existing: Self):
        pass

    fn get_gradients(self, x:MV, dloss_dy:MV) -> MV:
        pass

    fn update_weights(inout self, dif: MV):
        pass

