from kamo import dtype, simd_width
from kamo.libs.monum import MoVector, MoMatrix, MoNum, PI

alias SD = Scalar[dtype]
alias MN = MoNum[dtype, simd_width]
alias MV = MoVector[dtype, simd_width]
alias MM = MoMatrix[dtype, simd_width]


struct ChebyshevPolynomial[degree: Int = 3](EdgeFunc):
    var x_bounds: List[SD]
    var n_func: Int
    var weights: MV
    var nodes: MV

    fn __init__(inout self, x_bounds: List[SD], n_func: Int):
        self.x_bounds = x_bounds
        self.n_func = n_func

        self.weights = MV(n_func, 1)  # MN.rand(n_func)
        self.nodes = MV(n_func)

        self.set_chebyshev_nodes()

    fn __copyinit__(inout self, existing: Self):
        self.x_bounds = existing.x_bounds
        self.n_func = existing.n_func
        self.weights = existing.weights
        self.nodes = existing.nodes

    fn __moveinit__(inout self, owned existing: Self):
        self.x_bounds = existing.x_bounds^
        self.n_func = existing.n_func
        self.weights = existing.weights^
        self.nodes = existing.nodes^

    fn set_chebyshev_nodes(inout self):
        """Generate Chebyshev nodes within the given bounds."""

        var i = MN.arange(1, self.n_func + 1)
        var nodes = MN.cos((2 * i - 1.0) / (2.0 * self.n_func) * PI)
        self.nodes = 0.5 * (
            self.x_bounds[1] - self.x_bounds[0]
        ) * nodes + 0.5 * (self.x_bounds[1] + self.x_bounds[0])

    fn update_weights(inout self, dif: MV):
        self.weights += dif

    fn __call__(self, x: MV, grad: Bool = False) -> MV:
        var res = MV(len(x))

        if not grad:
            for i in range(self.n_func):
                res += self.weights[i] * self.chebyshev_polynomial(x, i)
        else:
            for i in range(self.n_func):
                res += self.weights[i] * self.chebyshev_polynomial_derivative(
                    x, i
                )

        return res

    fn chebyshev_polynomial(self, x: MV, n: Int) -> MV:
        if n == 0:
            return MV(len(x), 1.0)
        elif n == 1:
            return x
        else:
            var vec_minus_two = MV(len(x), 1.0)
            var vec_minus_one = x
            var res: MV = MV(len(x))
            for k in range(2, n + 1):
                res = 2 * x * vec_minus_one - vec_minus_two
                vec_minus_two = vec_minus_one
                vec_minus_one = res
            return res

    fn chebyshev_polynomial_derivative(self, x: MV, n: Int) -> MV:
        if n == 0:
            return MV(len(x))
        elif n == 1:
            return MV(len(x), 1.0)
        else:
            var vec_minus_two = MV(len(x), 1.0)
            var vec_minus_one = x

            var vec_prime_minus_two = MV(len(x))
            var vec_prime_minus_one = MV(len(x), 1.0)

            var res = MV(len(x))
            var res_prime = MV(len(x))

            for k in range(2, n + 1):
                res = 2 * x * vec_minus_one - vec_minus_two
                res_prime = (
                    2 * vec_minus_one
                    + 2 * x * vec_prime_minus_one
                    - vec_prime_minus_two
                )
                vec_minus_two = vec_minus_one
                vec_minus_one = res
                vec_prime_minus_two = vec_prime_minus_one
                vec_prime_minus_one = res_prime
            return res_prime
