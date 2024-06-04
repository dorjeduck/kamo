from math import abs

from kamo import dtype, simd_width
from kamo.func import silu
from kamo.libs.monum import MoVector, MoMatrix, MoNum

alias SD = Scalar[dtype]
alias MN = MoNum[dtype, simd_width]
alias MV = MoVector[dtype, simd_width]
alias MM = MoMatrix[dtype, simd_width]

alias tolerance_denominators = 1e-10

struct BSplineSilu[SPLINE_DEGREE: Int = 3,ADD_SILU:Bool=True](EdgeFunc):
    var x_bounds: List[SD]
    var num_trainable_params: Int
    var weights: MV
    var knots: MV

    fn __init__(inout self, x_bounds: List[SD], num_trainable_params: Int):
        self.x_bounds = x_bounds
        self.num_trainable_params = num_trainable_params

        self.weights = MN.rand(num_trainable_params)
        self.knots = MV(num_trainable_params+SPLINE_DEGREE+1)

        self._set_uniform_knots()

    fn __copyinit__(inout self, existing: Self):
        self.x_bounds = existing.x_bounds
        self.num_trainable_params = existing.num_trainable_params
        self.weights = existing.weights
        self.knots = existing.knots

    fn __moveinit__(inout self, owned existing: Self):
        self.x_bounds = existing.x_bounds^
        self.num_trainable_params = existing.num_trainable_params
        self.weights = existing.weights^
        self.knots = existing.knots^

    fn __del__(owned self):
        pass
        
    fn __call__(self, x: MV, grad: Bool = False) -> MV:
       
        var res = MV(len(x))
        var start = 1 if ADD_SILU else 0

        if not grad:    
            if ADD_SILU:
                res += self.weights[0]*silu(x)
           
            for i in range(start,self.weights.size-1):
                var b = self.basis_function(i, SPLINE_DEGREE, x)
                res += b * self.weights[i]
        else:
            if ADD_SILU:
                res += self.weights[0]*silu(x,True)
            for i in range(start,self.weights.size):
                var b_derivative = self.basis_function_derivative(
                    i, SPLINE_DEGREE, x
                )
                res += b_derivative * self.weights[i]

        return res

   
    fn update_weights(inout self, dif: MV):
        self.weights += dif

    fn calc_gradients(self, x:MV, dloss_dy:MV) -> MV:

        var gradients = MV(self.num_trainable_params)
        for i in range(self.num_trainable_params):
            gradients[i] = MN.sum(dloss_dy * self.basis_function(i, SPLINE_DEGREE, x))/x.size
        
        return gradients

    fn basis_function(self, i: Int, k: Int, x: MV) -> MV:
       
        var res = MV(x.size)

        if k == 0:
            for nn in range(x.size):
                if i >= len(self.knots) - SPLINE_DEGREE-2: 
                    if self.knots[i] <= x[nn] and x[nn] <= self.knots[i + 1]:
                        res[nn] = 1.0
                    else:
                        res[nn] = 0.0
                else:
                    if self.knots[i] <= x[nn] and x[nn] < self.knots[i + 1]:
                        res[nn] = 1.0
                    else:
                        res[nn] = 0.0
                    
            return res
        else:
            var denom1 = self.knots[i + k] - self.knots[i]
            var denom2 = self.knots[i + k + 1] - self.knots[i + 1]

            var term1: MV
            var term2: MV

            if abs(denom1) < tolerance_denominators:
                term1 = MV(x.size)
            else:
                term1 = (
                    (x - self.knots[i])
                    / denom1
                    * self.basis_function(i, k - 1, x)
                )
            if abs(denom2) < tolerance_denominators:
                term2 = MV(x.size)
            else:
                term2 = (
                    (self.knots[i + k + 1] - x)
                    / denom2
                    * self.basis_function(i + 1, k - 1, x)
                )
            return term1 + term2

    fn basis_function_derivative(self, i: Int, k: Int, x: MV) -> MV:
        """Derivative of B-spline basis function."""

        if k == 0:
            return MV(x.size, 0.0)
        else:
            var denom1 = self.knots[i + k] - self.knots[i]
            var denom2 = self.knots[i + k + 1] - self.knots[i + 1]

            var term1: MV
            var term2: MV

            if abs(denom1) < tolerance_denominators:
                term1 = MV(x.size)
            else:
                term1 = k / denom1 * self.basis_function(i, k - 1, x)

            if abs(denom2) < tolerance_denominators:
                term2 = MV(x.size)
            else:
                term2 = k / denom2 * self.basis_function(i + 1, k - 1, x)
            
            return term1 - term2

    fn _set_uniform_knots(inout self):
        var num_knots = len(self.knots)

        for i in range(SPLINE_DEGREE):
            self.knots[i] = self.x_bounds[0]

        var n_mid = num_knots - 2*(SPLINE_DEGREE)
       
        var step = (self.x_bounds[1] - self.x_bounds[0]) / (n_mid - 1)

        for i in range(n_mid ):
            self.knots[SPLINE_DEGREE +i] = self.x_bounds[0] + i * step
        self.knots[SPLINE_DEGREE + n_mid] = self.x_bounds[1]

        for i in range(SPLINE_DEGREE):
            self.knots[num_knots - 1 - i] = self.x_bounds[1]

    


