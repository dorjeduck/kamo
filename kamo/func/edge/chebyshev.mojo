from kamo import dtype, simd_width
from kamo.libs.monum import MoVector, MoMatrix, MoNum, PI

alias SD = Scalar[dtype]
alias MN = MoNum[dtype, simd_width]
alias MV = MoVector[dtype, simd_width]
alias MM = MoMatrix[dtype, simd_width]


struct ChebyshevPolynomial(EdgeFunc):
    var x_bounds: List[SD]
    var num_trainable_params: Int
    var weights: MV
    var nodes: MV

    var func_out:MM
    var tmp_x_size:MV


    fn __init__(inout self, num_trainable_params: Int,x_bounds: List[SD]):
        self.x_bounds = x_bounds
        self.num_trainable_params = num_trainable_params

        self.weights = MV(num_trainable_params, 1)  # MN.rand(num_trainable_params)
        self.nodes = MV(num_trainable_params)

        self.func_out = MM(0,0)
        self.tmp_x_size = MV(0)

        self.set_chebyshev_nodes()

    fn __copyinit__(inout self, existing: Self):
        self.x_bounds = existing.x_bounds
        self.num_trainable_params = existing.num_trainable_params
        self.weights = existing.weights
        self.nodes = existing.nodes

        self.func_out = existing.func_out
        self.tmp_x_size = existing.tmp_x_size

    fn __moveinit__(inout self, owned existing: Self):
        self.x_bounds = existing.x_bounds^
        self.num_trainable_params = existing.num_trainable_params
        self.weights = existing.weights^
        self.nodes = existing.nodes^

        self.func_out = existing.func_out^
        self.tmp_x_size = existing.tmp_x_size^

    fn __del__(owned self):
        pass

    fn set_chebyshev_nodes(inout self):
        """Generate Chebyshev nodes within the interval [-1, 1]."""

        var i = MN.arange(1, self.num_trainable_params + 1)
        self.nodes = MN.cos((2.0 * i - 1.0) / (2.0 * self.num_trainable_params) * PI)
        

    fn scale_to_unit(self, x:MV)->MV:
        """Scale x to the unit interval [-1, 1]."""
        return 2.0 * (x - self.x_bounds[0]) / (self.x_bounds[1] - self.x_bounds[0]) - 1.0


    fn update_weights(inout self, dif: MV):
        
        self.weights += dif

    fn __call__(inout self, x: MV, grad: Bool = False) -> MV:
        var res = MV(x.size)

        if self.func_out.cols != x.size:
            self.func_out = MM(self.weights.size,x.size)
            self.tmp_x_size = MV(x.size)

        if not grad:
            for i in range(self.num_trainable_params):
                var c = self.chebyshev_polynomial(x, i)
                res += self.weights[i] * c
                self.func_out.insert(c,i*x.size)
        else:
            for i in range(self.num_trainable_params):
                res += self.weights[i] * self.chebyshev_polynomial_derivative(
                    x, i
                )
        return res

    fn chebyshev_polynomial(self, x: MV, n: Int) -> MV:
        if n == 0:
            return MV(x.size, 1.0)
        elif n == 1:
            return x
        else:
           
            var xx = self.scale_to_unit(x)
            var vec_minus_two = MV(xx.size, 1.0)
            var vec_minus_one = xx
            var res: MV = MV(xx.size)
            for k in range(2, n + 1):
                res = 2 * xx * vec_minus_one - vec_minus_two
                vec_minus_two = vec_minus_one
                vec_minus_one = res
            return res
            

    fn chebyshev_polynomial_derivative(self, x: MV, n: Int) -> MV:
        if n == 0:
            return MV(x.size)
        elif n == 1:
            return MV(x.size, 1.0)
        else:
            var vec_minus_two = MV(x.size, 1.0)
            var vec_minus_one = x

            var vec_prime_minus_two = MV(x.size)
            var vec_prime_minus_one = MV(x.size, 1.0)

            var res = MV(x.size)
            var res_prime = MV(x.size)

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

    fn calc_gradients(inout self, x:MV, dloss_dy:MV) -> MV:
        
        var gradients = MV(self.num_trainable_params)
        for i in range(self.num_trainable_params):
            #gradients[i] = MN.sum(dloss_dy * self.chebyshev_polynomial(x, i))/x.size

            self.func_out.get_row(i,self.tmp_x_size)
            gradients[i] = MN.sum(dloss_dy * self.tmp_x_size)/x.size

        return gradients