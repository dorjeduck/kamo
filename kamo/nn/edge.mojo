from kamo import dtype,simd_width
from kamo.libs.monum import MoVector
from kamo.func.edge import EdgeFunc

alias SD = Scalar[dtype]
alias MV = MoVector[dtype,simd_width]

struct Edge[EF:EdgeFunc]:

    var edge_func:EF
    
    fn __init__(inout self,num_trainable_params:Int,x_bounds:List[SD]= List[SD](-1,1),rand_weights:Bool=True) raises:
        self.edge_func = EF(num_trainable_params,x_bounds)
            
        
    @always_inline
    fn __copyinit__(inout self, other: Self):
        self.edge_func = other.edge_func
        
    @always_inline
    fn __moveinit__(inout self, owned other: Self):
        self.edge_func = other.edge_func^
       
    fn __call__(inout self,x:MV,grad:Bool=False)->MV:
        return self.edge_func(x,grad)

    fn calc_gradients(inout self, x:MV, dloss_dy:MV) -> MV:
        return self.edge_func.calc_gradients(x,dloss_dy)

    fn update_weights(inout self,dif:MV):
        self.edge_func.update_weights(dif)
        
    @staticmethod
    fn get_list(size:Int,num_trainable_params:Int,x_bounds:List[SD]= List[SD](-1,1)) raises -> List[Self]:
        var res = List[Self](capacity=size)
        for i in range(size):
            res.append(Self(x_bounds,num_trainable_params))
        return res
