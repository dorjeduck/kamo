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

    var neuron = Neuron[False](2)

    var x = MM(2,3,12.0)

    print(x)

    var y = neuron(x)

    print(y)

   