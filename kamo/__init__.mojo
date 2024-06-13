# settings
alias dtype = DType.float64
alias simd_width = 2 * simdwidthof[dtype]()
alias eps = 1e-12

# convenience ...
from kamo.libs.monum import MoNum,MoVector,MoMatrix

alias SD = Scalar[dtype]
alias SD2 = InlineArray[SD, 2]
alias MN = MoNum[dtype, simd_width]
alias MV = MoVector[dtype, simd_width]
alias MM = MoMatrix[dtype, simd_width]