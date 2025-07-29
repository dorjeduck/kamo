from kamo import MN, MM, MV, SD, SD2


trait NeuronType(Copyable & Movable):
    fn __init__(
        out self,
        id: NeuronID,
        n_in: Int,
        num_trainable_edge_params: Int,
        weights_range: SD2,
        calc_phi_mat: Bool = True,
    ):
        ...

    ## common

    fn __call__(mut self, xin: MV, phi_mat: MM, phi_mat_der: MM) -> SD:
        ...

    fn calc_dxout_dxin(self):
        ...

    fn calc_dxout_dw(self):
        ...

    fn update_dloss_dw_dbias(mut self, dloss_dxout: SD):
        ...

    fn gradient_descent(mut self, learning_rate: SD):
        ...

    fn get_dxout_dxin(self) -> MV:
        ...

    fn zero_dloss_dw(self):
        ...

    fn zero_dloss_dbias(mut self):
        ...

    # neuron type specific

    fn calc_xmid(self, phi_mat: MM):
        ...

    fn calc_xout(mut self):
        ...

    fn calc_dxout_dxmid(mut self):
        ...

    fn calc_dxmid_dw(mut self, phi_mat: MM):
        ...

    fn calc_dxmid_dxin(mut self, phi_der_mat: MM):
        ...

    fn calc_dxout_dbias(mut self):
        ...

    @staticmethod
    fn phi_caching_capable() -> Bool:
        ...
