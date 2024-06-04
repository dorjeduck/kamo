from mokan import dtype,simd_width
from mokan.monum import MoVector

from mokan.nn.func import relu

alias SD = Scalar[dtype]
alias MV = MoVector[dtype,simd_width]
alias ACF = fn(SD,Bool)->SD

struct Neuron:

    var n_in:Int
    var weights:MV
    var bias:SD
    var activation: ACF
   
    fn __init__(inout self, n_in:Int,activation:ACF = relu ):
        self.n_in = n_in  
        self.weights = MV.rand(n_in)
        self.bias = 0
        self.activation = activation
    
    fn __init__(inout self, weights:List[SD],bias:SD,activation:ACF = relu):
        self.n_in = len(weights)
        self.weights = MV(weights) # n. inputs 
        self.bias = bias
        self.activation = activation

    @always_inline
    fn __copyinit__(inout self, other: Self):
        self.n_in = other.n_in
        self.weights = other.weights
        self.bias = other.bias 
        self.activation = other.activation

    @always_inline
    fn __moveinit__(inout self, owned other: Self):
        self.n_in = other.n_in
        self.weights = other.weights^
        self.bias = other.bias 
        self.activation = other.activation
       
    fn __call__(self,x:MV,grad:Bool=False)->SD:
        return self.activation(x @ self.weights + self.bias,grad)


    fn calc_gradients(self, x:MV, dloss_dy:MV) -> MV:
        return MV()
        #return self.edge_func.calc_gradients(x,dloss_dy)

    fn update_weights(inout self,dif:MV):
        pass
        #self.edge_func.update_weights(dif)

    @staticmethod
    fn get_list(size:Int,n_in:Int,activation:ACF=relu) -> List[Self]:
        var res = List[Self](capacity=size)
        for i in range(size):
            res.append(Self(n_in,activation))
        return res
