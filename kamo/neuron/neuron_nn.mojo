from kamo import MN, MM, MV, SD, SD2
from kamo.func import relu, ACF
from kamo.func.edge import EdgeFunc, BSplineSilu
from kamo.neuron import NeuronType


@value
struct NeuronNN[ACTIVATION_FUNC: ACF = relu](NeuronType):
    var n_in: Int  # number edges
    var xin: MV  # edges input
    var xmid: MV  # edges output, neuron input
    var weights: MM  # neuron weights
    var bias: SD  # neuron bias
    var xout: SD  # neuron output

    var dxout_dxmid: MV  # derivative d xout / d xmid:
    var dxout_dbias: SD  # derivative d xout / d bias
    var dxmid_dw: MM  # derivative d xmid / d w: (n_in, n_weights_per_edge)
    var dxmid_dxin: MV  # derivative d xmid / d xin
    var dxout_dxin: MV  # (composite) derivative d xout / d xin
    var dxout_dw: MM  # (composite) derivative d xout / d w
    var dloss_dw: MM  # (composite) derivative d loss / d w
    var dloss_dbias: SD  # (composite) derivative d loss / d bias

    var activation_input: SD

    var unit_vec: MV

    var id: NeuronID

    fn __init__(
        out self,
        id: NeuronID,
        n_in: Int,
        num_trainable_edge_params: Int,  # always 1 for NeuronNN
        weights_range: SD2,
        calc_phi_mat: Bool = True,  # not used by NN
    ):
        self.id = id
        self.n_in = n_in
        self.weights = MM.rand(
            self.n_in,
            1,  # one weight per edge
            weights_range[0],
            weights_range[1],
        )

        # self.weights = MM(self.n_in,1)

        self.xin = MV(self.n_in)
        self.xmid = MV(self.n_in)
        self.xout = 0

        self.bias = 0

        self.dxout_dxmid = MV(self.n_in)
        self.dxout_dbias = 0
        self.dxmid_dw = MM(self.n_in, 1)
        self.dxmid_dxin = MV(self.n_in)
        self.dxout_dxin = MV(self.n_in)
        self.dxout_dw = MM(self.n_in, 1)
        self.dloss_dw = MM(self.n_in, 1)
        self.dloss_dbias = 0

        self.activation_input = 0

        self.unit_vec = MV(self.n_in, 1.0)

    ## common to all neuron types

    fn __call__(mut self, xin: MV, phi_mat: MM, phi_der_mat: MM) -> SD:
        MN.inplace_copy(self.xin, xin)

        # forward pass: compute neuron's output
        self.calc_xmid(phi_mat)
        self.calc_xout()

        # compute internal derivatives
        self.calc_dxout_dxmid()
        self.calc_dxout_dbias()
        self.calc_dxmid_dw(phi_mat)
        self.calc_dxmid_dxin(phi_der_mat)

        # compute external derivativesdxout_dxin
        self.calc_dxout_dxin()
        self.calc_dxout_dw()

        # return neuron output
        return self.xout

    fn calc_dxout_dxin(self):
        MN.inplace_copy(self.dxout_dxin, self.dxout_dxmid * self.dxmid_dxin)

    fn get_dxout_dxin(self) -> MV:
        return self.dxout_dxin

    fn calc_dxout_dw(self):
        MN.inplace_copy(
            self.dxout_dw, MN.diag(self.dxout_dxmid) @ self.dxmid_dw
        )

    fn update_dloss_dw_dbias(mut self, dloss_dxout: SD):
        self.dloss_dw += self.dxout_dw * dloss_dxout
        self.dloss_dbias += self.dxout_dbias * dloss_dxout

    fn gradient_descent(mut self, learning_rate: SD):
        self.weights -= learning_rate * self.dloss_dw
        self.bias -= learning_rate * self.dloss_dbias

    fn zero_dloss_dw(self):
        self.dloss_dw.zero()

    fn zero_dloss_dbias(mut self):
        self.dloss_dbias = 0.0

    ## nn specific

    fn calc_xmid(self, inoutphi_mat: MM):
        MN.inplace_copy(self.xmid, self.weights.get_col(0) * self.xin)

    fn calc_xout(mut self):
        self.activation_input = MN.sum(self.xmid) + self.bias
        self.xout = ACTIVATION_FUNC(self.activation_input)

    fn calc_dxout_dxmid(mut self):
        self.dxout_dxmid = (
            ACTIVATION_FUNC(self.activation_input, True) * self.unit_vec
        )

    fn calc_dxout_dbias(mut self):
        self.dxout_dbias = ACTIVATION_FUNC(self.activation_input, True)

    fn calc_dxmid_dw(mut self, phi_mat: MM):
        self.dxmid_dw = MM(self.xin.size, 1, self.xin)

    fn calc_dxmid_dxin(mut self, phi_der_mat: MM):
        self.dxmid_dxin = self.weights.flatten()

    fn __del__(owned self):
        pass

    @staticmethod
    fn phi_caching_capable() -> Bool:
        return False
