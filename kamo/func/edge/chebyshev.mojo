from kamo import dtype, simd_width
from kamo.libs.monum import MoVector, MoMatrix, MoNum, PI

alias SD = Scalar[dtype]
alias MN = MoNum[dtype, simd_width]
alias MV = MoVector[dtype, simd_width]
alias MM = MoMatrix[dtype, simd_width]
alias SD2 = InlineArray[SD,2]

@value
struct ChebyshevPolynomial(EdgeFunc):
    var x_bounds: SD2
    var num_trainable_params: Int
    var nodes: MV

    fn __init__(inout self, num_trainable_params: Int,x_bounds: SD2=SD2(-1,1)):
        self.x_bounds = x_bounds
        self.num_trainable_params = num_trainable_params
        self.nodes = MV(num_trainable_params)
        
        self.set_chebyshev_nodes()


    fn __del__(owned self):
        pass

    fn set_chebyshev_nodes(inout self):
        """Generate Chebyshev nodes within the interval [-1, 1]."""

        var i = MN.arange(1, self.num_trainable_params + 1)
        self.nodes = MN.cos((2.0 * i - 1.0) / (2.0 * self.num_trainable_params).cast[dtype]() * PI)
        

    fn scale_to_unit(self, x:MV)->MV:
        """Scale x to the unit interval [-1, 1]."""
        return 2.0 * (x - self.x_bounds[0]) / (self.x_bounds[1] - self.x_bounds[0]) - 1.0


   
    fn calc_phi_mat(inout self,inout phi_mat:MM, x: MV, grad: Bool = False):
        
        if not grad:
            for i in range(self.num_trainable_params):
                var c = self.chebyshev_polynomial(x, i)     
                phi_mat.insert_row(i,c)
        else:
            for i in range(self.num_trainable_params):
                var c = self.chebyshev_polynomial_derivative(
                    x, i
                )
                phi_mat.insert_row(i,c)
       

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
