from algorithm import vectorize
from sys.info import simdwidthof
from math import mul, div, mod, pow,add, trunc, align_down, align_down_residual

from kamo import dtype,simd_width 
from kamo.monum import MoVector,MoMatrix,MoNum

alias SD = Scalar[dtype]
alias MN = MoNum[dtype,simd_width]
alias MV = MoVector[dtype,simd_width]
alias MM = MoMatrix[dtype,simd_width]

###############
## chebyshev ##
###############

fn chebyshev_polynomial(n:Int, x:MV) -> MV:
    
    if n == 0:
        return MV(x.size,1.0)
    elif n == 1:
        return x
    else:
       
        var T0 = MV(x.size,1.0)
        var T1 = x
        
        for _ in range(2, n + 1):
            var T2 = 2 * x * T1 - T0
            T0 = T1
            T1 = T2
        return T1
      
fn chebyshev_derivative(n:Int, x:MV) -> MV:
    
    if n == 0:
        return  MV(x.size)
    elif n == 1:
        return  MV(x.size,1.0)
    else:
        var U0 = MV(x.size,1.0)
        var U1 = 2 * x
        for _ in range(2, n):
            var U2 = 2 * x * U1 - U0
            U0 = U1
            U1 = U2
        return n * U1
       
fn create_chebyshev_basis_function( deg:Int, x_bounds:List[SD]) -> fn (MV) escaping -> MV:

    fn basis_function(x:MV) -> MV:
        var normalized_x = (2 * ( x - x_bounds[0]) / (x_bounds[1] - x_bounds[0])) - 1.0
        return chebyshev_polynomial(deg, normalized_x)
       
    return basis_function

fn create_chebyshev_derivative_function( deg:Int, x_bounds:List[SD]) -> fn (MV) escaping -> MV:
      
    fn derivative_function(x:MV) -> MV:
        var normalized_x = (2 * (x - x_bounds[0]) / (x_bounds[1] - x_bounds[0])) - 1.0
        return chebyshev_derivative(deg, normalized_x) * (2 / (x_bounds[1] - x_bounds[0]))
    
    return derivative_function

fn get_weighted_chebyshev(x_bounds:List[SD], n_func:Int) raises ->
    fn (x:MV,w:MV,/,grad:Bool=False) escaping -> MV : 
 
    var edge_func = List[fn(x:MV,/,grad:Bool=False) escaping -> MV](capacity=n_func)
   
    for deg in range(n_func):
        var _f = create_chebyshev_basis_function(deg, x_bounds)
        var _f_der = create_chebyshev_derivative_function(deg, x_bounds)

        fn f ( x:MV,/,grad:Bool=False) escaping -> MV:
            if not grad:
                return _f(x)
            else:
                return _f_der(x)

        edge_func.append(f)
    
    fn fres( x:MV, w:MV,/,grad:Bool=False) escaping -> MV: 
        
        var n_in = len(x)
        var mat = MM(n_func,n_in)
        for i in range(n_func):
            mat.insert(i*n_in,edge_func[i](x,grad))
        var vec = w @ mat
        return vec

    return fres

############## 
## bsplines ##
##############

fn bspline_basis_element(i:Int, k:Int, t:MV, x:MV) -> MV:
   
    var res = MV(x.size)
    
    if k == 0:
       
        for nn in range(x.size):
            if t[i] <= x[nn] and x[nn] < t[i+1]:
                res[nn]=1.0
            else:
                res[nn]=0.0
    else:
        
        var denom1 = t[i + k] - t[i]
        var denom2 = t[i + k + 1] - t[i + 1]
        var term1:MV
        var term2:MV
        if denom1 == 0:
            term1 = MV(x.size)
        else:
            term1 = (x - t[i]) / denom1 * bspline_basis_element(i, k - 1, t, x)
        if denom2 == 0:
            term2 = MV(x.size)
        else:
            term2 = (t[i + k + 1] - x) / denom2 * bspline_basis_element(i + 1, k - 1, t, x)
        res = term1 + term2
       
    return res

fn bspline_derivative_element(i:Int, k:Int, t:MV, x:MV) -> MV:
    if k == 0:
        return MV(x.size,0.0)

    var denom1 = t[i + k] - t[i]
    var denom2 = t[i + k + 1] - t[i + 1]

    var term1:MV
    var term2:MV

    if denom1 == 0:
        term1 = MV(x.size)
    else:
        term1 = k / denom1 * bspline_basis_element(i, k - 1, t, x)

    if denom2 == 0:
        term2 = MV(x.size)
    else:
        term2 = k / denom2 * bspline_basis_element(i + 1, k - 1, t, x)
    
    return term1 - term2
   
fn create_bspline_basis_function(ind_spline:Int, degree:Int, t:MV) -> fn (MV) escaping -> MV:
      
    fn _f(x:MV) -> MV:
        return bspline_basis_element(ind_spline, degree, t, x)
    return _f 

fn create_bspline_derivative_function(ind_spline:Int, degree:Int, t:MV) -> fn (MV) escaping -> MV:
   
    fn _f(x:MV) -> MV:
        return bspline_derivative_element(ind_spline, degree, t, x)
    return _f 
    
fn get_weighted_bsplines(x_bounds:List[SD], n_func:Int, degree:Int=3) raises ->
    fn (x:MV,w:MV,/,grad:Bool=False) escaping -> MV : 
    
    var grid_len = n_func - degree + 1
    var step = (x_bounds[1] - x_bounds[0]) / (grid_len - 1)
    var edge_func = List[fn(x:MV,/,grad:Bool=False) escaping -> MV](capacity=n_func)
   
    # SiLU bias function
    fn _silu_bias(x:MV) escaping -> MV :
        return  x / (1 + (-x).exp())
       
    fn _silu_bias_der(x:MV) escaping -> MV :
        return  (1 + (-x).exp() + x * (-x).exp()) / ((1 + (-x).exp()) ** 2)
       
    fn _silu (x:MV,/,grad:Bool=False) escaping -> MV:
        if not grad:
            return _silu_bias(x)
        else:
            return _silu_bias_der(x)

    edge_func.append(_silu)
    
    # Create knot vector
    var t = MN.linspace(x_bounds[0] - degree * step, x_bounds[1] + degree * step, grid_len + 2 * degree)
    t[degree] = x_bounds[0]
    t[-degree - 1] = x_bounds[1]
    
    # Generate B-splines and their derivatives
    for ind_spline in range(n_func - 1):
        
        var _f = create_bspline_basis_function(ind_spline, degree, t)
        var _f_der = create_bspline_derivative_function(ind_spline, degree, t)
        
        fn f (x:MV,/,grad:Bool=False) escaping -> MV:
            if not grad:
                return _f(x)
            else:
                return _f_der(x)

        edge_func.append(f)

    fn fres(x:MV,w:MV,/,grad:Bool=False) escaping -> MV: 
        
        var n_in = len(x)
        var mat = MM(n_func,n_in)
        for i in range(n_func):
            mat.insert(i*n_in,edge_func[i](x,grad))
           

        var vec = w @ mat
        return vec

    return fres