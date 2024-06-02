from kamo import dtype,simd_width
from kamo.monum import MoVector

alias SD = Scalar[dtype]
alias MV = MoVector[dtype,simd_width]
alias EF = fn ( x:MV, w:MV,/,grad:Bool=False) escaping -> MV

struct Edge:

    var n_weights:Int
    var edge_func:EF
    var weights:MV
    
    fn __init__(inout self,edge_func:EF, n_weights:Int,rand_weights:Bool=True):
        self.edge_func = edge_func
        self.n_weights = n_weights
        if rand_weights:
            self.weights = MV.rand(self.n_weights)
        else:
            self.weights = MV(self.n_weights)
      

        
      
        
    fn __init__(inout self,edge_func:EF,weights:MV):
        self.edge_func = edge_func
        self.n_weights = len(weights)
        self.weights = weights 
       
        
    @always_inline
    fn __copyinit__(inout self, other: Self):
        self.edge_func = other.edge_func
        self.n_weights = other.n_weights
        self.weights = other.weights
       
       
    @always_inline
    fn __moveinit__(inout self, owned other: Self):
        self.edge_func = other.edge_func
        self.n_weights = other.n_weights
        self.weights = other.weights^
       
    fn __call__(inout self,x:MV,grad:Bool=False)->MV:
        return self.edge_func(x,self.weights,grad)
        
    @staticmethod
    fn get_list(size:Int,edge_func:EF, n_weights:Int) -> List[Self]:
        var res = List[Self](capacity=size)
        for i in range(size):
            res.append(Self(edge_func,n_weights))
        return res
