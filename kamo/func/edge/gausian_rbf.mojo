from sys.terminate import exit

from kamo import MN, MM, MV, SD, SD2


@value
struct GaussianRBF(EdgeFunc):
    var x_bounds: SD2
    var num_trainable_params: Int
    var centers: MV
    var sigma: SD

    fn __init__(
        out self, num_trainable_params: Int, x_bounds: SD2 = SD2(-1, 1)
    ):
        self.x_bounds = x_bounds
        self.num_trainable_params = num_trainable_params
        self.centers = MV(num_trainable_params)
        self.sigma = 0  # Mojo requires this
        self.set_uniform_centers()

        # Set sigma based on center spacing
        self.sigma = (self.centers[1] - self.centers[0]) / 2

    fn __del__(owned self):
        pass

    fn calc_phi_mat(self, phi_mat: MM, xx: MV, grad: Bool = False):
        var x = self.scale_to_unit(xx)
        if not grad:
            for i in range(self.num_trainable_params):
                var b = self.basis_function(i, x)
                phi_mat.insert_row(i, b)
        else:
            for i in range(self.num_trainable_params):
                var b = self.basis_function_derivative(i, x)
                phi_mat.insert_row(i, b)

    fn basis_function(self, i: Int, x: MV) -> MV:
        var res = MV(x.size)
        var center = self.centers[i]
        for nn in range(x.size):
            res[nn] = MN.exp(
                -((x[nn] - center) * (x[nn] - center))
                / (2 * self.sigma * self.sigma)
            )
        return res

    fn basis_function_derivative(self, i: Int, x: MV) -> MV:
        var res = MV(x.size)
        var center = self.centers[i]
        for nn in range(x.size):
            var value = MN.exp(
                -((x[nn] - center) * (x[nn] - center))
                / (2 * self.sigma * self.sigma)
            )
            res[nn] = -((x[nn] - center) / (self.sigma * self.sigma)) * value
        return res

    @always_inline
    fn scale_to_unit(self, x: MV) -> MV:
        """Scale x to the unit interval [-1, 1]."""
        return (
            2.0 * (x - self.x_bounds[0]) / (self.x_bounds[1] - self.x_bounds[0])
            - 1.0
        )

    fn set_uniform_centers(self):
        var num_centers = len(self.centers)
        var step = (self.x_bounds[1] - self.x_bounds[0]) / (num_centers - 1)
        for i in range(num_centers):
            self.centers[i] = self.x_bounds[0] + i * step
