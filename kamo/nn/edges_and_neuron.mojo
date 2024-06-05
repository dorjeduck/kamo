
from kamo.nn import Edge,Neuron

from kamo.libs.monum import MoVector,MoMatrix
from kamo.func.edge import EdgeFunc

alias SD = Scalar[dtype]
alias MV = MoVector[dtype,simd_width]
alias MM = MoMatrix[dtype,simd_width]

struct EdgesAndNeuron[EF:EdgeFunc]:

    var edges:List[Edge[EF]]
    var neuron:Neuron
    
    
    fn __init__(inout self,x_bounds:List[SD],n_edges:Int) raises:

        self.edges = Edge[EF].get_list(x_bounds,n_edges,11)
        self.neuron = Neuron(n_edges)


    fn __call__(self,x:MM):
        
