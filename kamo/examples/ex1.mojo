from random.random import seed
from time import now 

from kamo import dtype,simd_width
from kamo.libs.monum import MoVector
from kamo.nn import Edge
from kamo.func.edge import BSpline

alias SD = Scalar[dtype]
alias MV = MoVector[dtype,simd_width]

fn main() raises:

    seed(now())

    var x_bounds = List[SD](0,10)
    var weights = MV(List[SD](1,1,1))

    var num_trainable_params = 11
    
    var edge = Edge[BSpline[3]](x_bounds,num_trainable_params)

    var x = MV(List[SD](1.,2.,10.))

    print(edge(x))
    print(edge(x,True))

