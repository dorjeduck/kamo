
from python import Python

from kamo import MN,MM,SD2
from kamo.func.edge import BSplineSilu,ChebyshevPolynomial,GaussianRBF
from kamo.utils import BasisFunctionPlotter

fn main() raises:
    var num_func = 8
    var num_plot_data = 200
   
    var spline = BSplineSilu[3](num_func)
    var chebyshev = ChebyshevPolynomial(num_func)
    var gauss = GaussianRBF(num_func)

    ## plot

    var bfp = BasisFunctionPlotter(num_func,num_plot_data)
    
    bfp.plot[BSplineSilu[3]](spline,"Cubic B-Spline & SILU basis functions","imgs/bspline_silu_basis.png")
    bfp.plot[BSplineSilu[3]](spline,"Derivatives of the Cubic B-Spline & SILU basis functions","imgs/bspline_silu_basis_der.png",True)
    bfp.plot[ChebyshevPolynomial](chebyshev,"Chebyshev Polynomials basis functions","imgs/chebyshev_basis.png")
    bfp.plot[ChebyshevPolynomial](chebyshev,"Derivatives of the Chebyshev Polynomials basis functions","imgs/chebyshev_basis_der.png",True)
    bfp.plot[GaussianRBF](gauss,"Gaussian radial basis functions","imgs/gaussian_rbf.png")
    bfp.plot[GaussianRBF](gauss,"Derivatives of the Gaussian radial basis functions","imgs/gaussian_rbf_der.png",True)
    
    