from algorithm import vectorize

from kamo import dtype, simd_width
from kamo.func import silu
from kamo.libs.monum import MoVector, MoMatrix, MoNum

alias SD = Scalar[dtype]
alias MV = MoVector[dtype, simd_width]
alias MM = MoMatrix[dtype, simd_width]
alias SD2 = InlineArray[SD,2]

@value
struct Identity(EdgeFunc):
    var num_trainable_params: Int
    
    fn __init__(inout self, num_trainable_params: Int,x_bounds: SD2):
        self.num_trainable_params = num_trainable_params
       
    fn __del__(owned self):
        pass

    fn calc_phi_mat(inout self, inout phi_mat:MM, xx: MV, grad: Bool = False):
       
        if not grad:  
            for i in range(self.num_trainable_params):
                phi_mat.insert_col(i,xx)
        else:
            @parameter
            fn _one[width:Int](iv:Int):
                phi_mat._mat_ptr.store[width=width](iv,1.0)
            vectorize[_one,simd_width](self.num_trainable_params*xx.size)
        