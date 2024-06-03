from kamo import dtype, simd_width
from kamo.libs.monum import MoVector, MoMatrix, MoNum

alias SD = Scalar[dtype]
alias MN = MoNum[dtype, simd_width]
alias MV = MoVector[dtype, simd_width]
alias MM = MoMatrix[dtype, simd_width]


struct BSpline[degree: Int = 3](EdgeFunc):
    var x_bounds: List[SD]
    var n_func: Int
    var weights: MV
    var knots: MV

    fn __init__(inout self, x_bounds: List[SD], n_func: Int):
        self.x_bounds = x_bounds
        self.n_func = n_func

        self.weights = MN.rand(n_func)
        self.knots = MV(n_func)

        self.set_uniform_knots()

    fn set_uniform_knots(inout self):
        var num_knots = len(self.knots)

        for i in range(self.degree):
            self.knots[i] = self.x_bounds[0]

        var n_mid = num_knots - 2 * self.degree
        var step = (self.x_bounds[1] - self.x_bounds[0]) / (n_mid - 1)

        for i in range(n_mid - 1):
            self.knots[self.degree + i] = self.x_bounds[0] + i * step
        self.knots[self.degree + n_mid] = self.x_bounds[1]

        for i in range(self.degree):
            self.knots[num_knots - 1 - i] = self.x_bounds[1]

    fn __copyinit__(inout self, existing: Self):
        self.x_bounds = existing.x_bounds
        self.n_func = existing.n_func
        self.weights = existing.weights
        self.knots = existing.knots

    fn __moveinit__(inout self, owned existing: Self):
        self.x_bounds = existing.x_bounds^
        self.n_func = existing.n_func
        self.weights = existing.weights^
        self.knots = existing.knots^

    fn __del__(owned self):
        ##todo
        pass

    fn update_weights(inout self, dif: MV):
        self.weights += dif

    fn basis_function(self, i: Int, k: Int, x: MV) -> MV:
        """Recursive definition of B-spline basis function."""

        var res = MV(x.size)

        if k == 0:
            for nn in range(x.size):
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

            if denom1 == 0:
                term1 = MV(x.size)
            else:
                term1 = (
                    (x - self.knots[i])
                    / denom1
                    * self.basis_function(i, k - 1, x)
                )
            if denom2 == 0:
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

            if denom1 == 0:
                term1 = MV(x.size)
            else:
                term1 = k / denom1 * self.basis_function(i, k - 1, x)

            if denom2 == 0:
                term2 = MV(x.size)
            else:
                term2 = k / denom2 * self.basis_function(i + 1, k - 1, x)

            return term1 - term2

    fn __call__(self, x: MV, grad: Bool = False) -> MV:
        """
        Evaluate the B-spline or the derivative of the B-spline
        at given points x (batch processing).
        """

        var res = MV(len(x))

        if not grad:
            for i in range(len(self.weights)):
                var b = self.basis_function(i, degree, x)

                res += b * self.weights[i]

        else:
            for i in range(len(self.weights)):
                var b_derivative = self.basis_function_derivative(
                    i, self.degree, x
                )
                res += b_derivative * self.weights[i]

        return res
