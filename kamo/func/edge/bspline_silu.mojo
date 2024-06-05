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

    var func_out:MM
    var tmp_x_size:MV



    fn __init__(inout self, num_trainable_params: Int,x_bounds: List[SD]):
        self.x_bounds = x_bounds
        self.num_trainable_params = num_trainable_params

        self.weights = MN.rand(num_trainable_params)
        self.knots = MV(num_trainable_params+SPLINE_DEGREE+1)

        self.func_out = MM(0,0)
        self.tmp_x_size = MV(0)
        
        self.set_uniform_knots()



    fn __copyinit__(inout self, existing: Self):
        self.x_bounds = existing.x_bounds
        self.num_trainable_params = existing.num_trainable_params
        self.weights = existing.weights
        self.knots = existing.knots
        self.func_out = existing.func_out
        self.tmp_x_size = existing.tmp_x_size

    fn __moveinit__(inout self, owned existing: Self):
        self.x_bounds = existing.x_bounds^
        self.num_trainable_params = existing.num_trainable_params
        self.weights = existing.weights^
        self.knots = existing.knots^
        self.func_out = existing.func_out^
        self.tmp_x_size = existing.tmp_x_size^

    fn __del__(owned self):
        pass

    fn __call__(inout self, xx: MV, grad: Bool = False) -> MV:
        var x = self.scale_to_unit(xx)

        if self.func_out.cols != x.size:
            self.func_out = MM(self.weights.size,x.size)
            self.tmp_x_size = MV(x.size)
            
       
        var res = MV(x.size)
        var start = 1 if ADD_SILU else 0

        if not grad:    
            if ADD_SILU:
                var s = silu(x)
                res += self.weights[0]*s
                self.func_out.insert(s,0)
           
            for i in range(start,self.weights.size-1):
                var b = self.basis_function(i, SPLINE_DEGREE, x)
                res += b * self.weights[i]
                self.func_out.insert(b,i*x.size)
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

    fn calc_gradients(inout self, xx:MV, dloss_dy:MV) -> MV:
        var x = self.scale_to_unit(xx)
        var gradients = MV(self.num_trainable_params)

        for i in range(self.num_trainable_params):
            #gradients[i] = MN.sum(dloss_dy * self.basis_function(i, SPLINE_DEGREE, x))/x.size
            
            self.func_out.get_row(i,self.tmp_x_size)
            gradients[i] = MN.sum(dloss_dy * self.tmp_x_size)/x.size
            
        return gradients

    fn basis_function(self, i: Int, k: Int, x: MV) -> MV:
       
        var res = MV(x.size)

        if k == 0:
            for nn in range(x.size):
                if i >= self.knots.size - SPLINE_DEGREE-2: 
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

    @always_inline
    fn scale_to_unit(self, x:MV)->MV:
        """Scale x to the unit interval [-1, 1]."""
        return 2.0 * (x - self.x_bounds[0]) / (self.x_bounds[1] - self.x_bounds[0]) - 1.0


    fn set_uniform_knots(inout self):
        var num_knots = len(self.knots)

        for i in range(SPLINE_DEGREE):
            self.knots[i] = -1

        var n_mid = num_knots - 2*(SPLINE_DEGREE)
       
        var step = 2 / (n_mid - 1)

        for i in range(n_mid ):
            self.knots[SPLINE_DEGREE +i] = -1 + i * step
       
        for i in range(SPLINE_DEGREE+1):
            self.knots[num_knots - 1 - i] = 1

