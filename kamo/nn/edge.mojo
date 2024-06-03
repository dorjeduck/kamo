from kamo import dtype,simd_width
from kamo.libs.monum import MoVector
from kamo.nn.func.edge import EdgeFunc

alias SD = Scalar[dtype]
alias MV = MoVector[dtype,simd_width]

struct Edge[EF:EdgeFunc]:

    var edge_func:EF
    
    
    fn __init__(inout self,x_bounds:List[SD], n_func:Int,rand_weights:Bool=True) raises:
        self.edge_func = EF(x_bounds,n_func)
            
    fn __init__(inout self,x_bounds:List[SD],weights:MV) raises:
        self.edge_func = EF(x_bounds,len(weights))
        
    @always_inline
    fn __copyinit__(inout self, other: Self):
        self.edge_func = other.edge_func
        
    @always_inline
    fn __moveinit__(inout self, owned other: Self):
        self.edge_func = other.edge_func^
       
    fn __call__(inout self,x:MV,grad:Bool=False)->MV:
        return self.edge_func(x,grad)

    fn update_weights(inout self,dif:MV):
        self.edge_func.update_weights(dif)
        
    @staticmethod
    fn get_list(size:Int,edge_func:EF, n_weights:Int) -> List[Self[EF]]:
        var res = List[Self[EF]](capacity=size)
        for i in range(size):
            res.append(Self[EF](self.x_bounds,n_weights))
        return res
