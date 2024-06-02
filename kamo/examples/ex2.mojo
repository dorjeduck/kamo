from random.random import seed
from time import now 

from kamo import dtype,simd_width
from kamo.monum import MoNum,MoVector
from kamo.nn import Edge
from kamo.nn.func import get_weighted_bsplines

alias SD = Scalar[dtype]
alias MV = MoVector[dtype,simd_width]
alias MN = MoNum[dtype,simd_width]

fn main() raises:

    seed(now())

    var x_bounds = List[SD](0,10)

    # Training data
    var x_train = MN.linspace(x_bounds[0], x_bounds[1], 100)
    var y_train = MN.sin(x_train)

    var weights = MV(List[SD](1,1,1))

    var n_func = 5
    var degree = 3

    var edge_func = get_weighted_bsplines(x_bounds, n_func, degree)

    var edge = Edge(edge_func,n_func)

    var x = MV(List[SD](1.,2.,3.))

    print(edge(x))

