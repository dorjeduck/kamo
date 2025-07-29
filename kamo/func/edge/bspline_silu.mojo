from sys.terminate import exit 
from kamo import MN,MM,MV,SD,SD2
from kamo.func import silu_mv

alias tolerance_denominators = 1e-10

@value
struct BSplineSilu[SPLINE_DEGREE: Int = 3, ADD_SILU: Bool = True](EdgeFunc):
    var x_bounds: SD2
    var num_trainable_params: Int
    var knots: MV

    fn __init__(out self, num_trainable_params: Int, x_bounds: SD2=SD2(-1,1)):
        self.x_bounds = x_bounds
        self.num_trainable_params = num_trainable_params
        var start = 1 if ADD_SILU else 0
        self.knots = MV(num_trainable_params - start + SPLINE_DEGREE + 1)
        self.set_uniform_knots()

    fn __del__(owned self):
        pass

    fn calc_phi_mat(self, phi_mat: MM, xx: MV, grad: Bool = False):
        var x = self.scale_to_unit(xx)
        var start = 1 if ADD_SILU else 0

        if not grad:
            if ADD_SILU:
                var s = silu_mv(x)
                phi_mat.insert_row(0, s)

            for i in range(start, self.num_trainable_params):
                var b = self.basis_function(i - start, SPLINE_DEGREE, x)
                phi_mat.insert_row(i, b)
        else:
            if ADD_SILU:
                var s = silu_mv(x, True)
                MN.nantozero(s)
                phi_mat.insert_row(0, s)

            for i in range(start, self.num_trainable_params):
                var b = self.basis_function_derivative(
                    i - start, SPLINE_DEGREE, x
                )
                MN.nantozero(b)
                phi_mat.insert_row(i, b)

    fn basis_function(self, i: Int, k: Int, x: MV) -> MV:
        var res = MV(x.size)

        if k == 0:
            for nn in range(x.size):
                if i >= self.knots.size - SPLINE_DEGREE - 2:
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
    fn scale_to_unit(self, x: MV) -> MV:
        """Scale x to the unit interval [-1, 1]."""
        return (
            2.0 * (x - self.x_bounds[0]) / (self.x_bounds[1] - self.x_bounds[0])
            - 1.0
        )

    fn set_uniform_knots(self):
        var num_knots = len(self.knots)

        for i in range(SPLINE_DEGREE):
            self.knots[i] = -1

        var n_mid = num_knots - 2 * (SPLINE_DEGREE)

        var step = (2 / (n_mid - 1)).cast[dtype]()

        for i in range(n_mid):
            self.knots[SPLINE_DEGREE + i] = -1 + i * step

        for i in range(SPLINE_DEGREE + 1):
            self.knots[num_knots - 1 - i] = 1
