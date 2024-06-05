from random.random import seed
from time import now,sleep

from math import sin,cos

from kamo import dtype,simd_width
from kamo.libs.monum import MoNum,MoVector,MoMatrix,PI
from kamo.nn import Neuron,EdgesAndNeuron
from kamo.func import SquaredLoss
from kamo.func.edge import ChebyshevPolynomial

from kamo.libs.mopro import progress_bar,BarSettings


alias SD = Scalar[dtype]
alias MV = MoVector[dtype,simd_width]
alias MM = MoMatrix[dtype,simd_width]
alias MN = MoNum[dtype,simd_width]

fn main() raises:

    var n_edges = 2
    var train_samples = 4

    var x_train = MN.linspace2D(0,10,n_edges,train_samples)
    var y_train = MV(train_samples)

    for i in range(train_samples):
        y_train[i] = sin(x_train[0,i]) + cos(x_train[1,i])


    print(x_train)
    print(y_train)

    
    var x_bounds = List[SD](-1,1)
    

    var edges_and_neuron = EdgesAndNeuron[ChebyshevPolynomial](x_bounds,n_edges)

    
    