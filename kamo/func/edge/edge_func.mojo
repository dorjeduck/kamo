from kamo import MN, MM, MV, SD, SD2


trait EdgeFunc(Copyable, Movable):
    fn __init__(
        out self, num_trainable_params: Int, x_bounds: SD2 = SD2(-1, 1)
    ):
        pass

    fn calc_phi_mat(self, phi_mat: MM, x: MV, grad: Bool = False):
        pass
