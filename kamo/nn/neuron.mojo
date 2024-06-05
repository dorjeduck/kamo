from kamo import dtype,simd_width
from kamo.libs.monum import MoVector,MoMatrix,MoNum

from kamo.func import relu,tanh_act

alias SD = Scalar[dtype]
alias MM = MoMatrix[dtype,simd_width]
alias MN = MoNum[dtype,simd_width]
alias MV = MoVector[dtype,simd_width]
alias ACF = fn(SD,Bool)->SD

struct Neuron[Normalize:Bool=False]:

    var n_in:Int
   
    fn __init__(inout self, n_in:Int ):
        self.n_in = n_in  
       
    @always_inline
    fn __copyinit__(inout self, other: Self):
        self.n_in = other.n_in
      
    @always_inline
    fn __moveinit__(inout self, owned other: Self):
        self.n_in = other.n_in
        
    fn __call__(self,mx:MM,grad:Bool=False)->MV:
        @parameter
        if Normalize:
            if not grad:
                return tanh_act(MN.sum(mx,axis=0))
            else:
                return tanh_act(MN.sum(mx,axis=0), True) 
        else:
            if not grad:
                return MN.sum(mx,axis=0)
            else:
                return MV(mx.cols,1.0)


    #fn calc_gradients(self, x:MV, dloss_dy:MV) -> MV:
    #    return MV()
    #    #return self.edge_func.calc_gradients(x,dloss_dy)
    #
    #fn update_weights(inout self,dif:MV):
    #    pass
    #    #self.edge_func.update_weights(dif)
    #
    #@staticmethod
    #fn get_list(size:Int,n_in:Int,activation:ACF=relu) -> List[Self]:
    #    var res = List[Self](capacity=size)
    #    for i in range(size):
    #        res.append(Self(n_in,activation))
    #    return res
