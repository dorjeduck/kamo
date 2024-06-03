from random.random import seed
from time import now 

from kamo import dtype,simd_width
from kamo.libs.monum import MoVector
from kamo.nn import Edge
from kamo.nn.func.edge import ChebyshevPolynomial

alias SD = Scalar[dtype]
alias MV = MoVector[dtype,simd_width]

fn main() raises:

    seed(now())

    var x_bounds = List[SD](0,10)
    var weights = MV(List[SD](1,1,1))

    var n_func = 5
    
    var edge = Edge[ChebyshevPolynomial[4]](x_bounds,n_func)

    var x = MV(List[SD](0,5,10))

    print(edge(x))
    print(edge(x,True))

