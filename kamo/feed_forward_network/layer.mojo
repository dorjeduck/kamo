from kamo import MN, MM, MV, SD, SD2
from kamo.neuron import NeuronType, NeuronID


@value
struct FullyConnectedLayer[NT: NeuronType, PHI_CACHING: Bool](
    Copyable & Movable
):
    var n_in: Int
    var n_out: Int
    var neurons: List[NT]
    var xin: MV
    var xout: MV
    var dloss_dxin: MV

    var id: Int

    var phi_mat: MM
    var phi_der_mat: MM

    var do_phi_caching: Bool

    fn __init__(
        out self,
        id: Int,
        n_in: Int,
        n_out: Int,
        num_trainable_edge_params: Int,
        weights_range: InlineArray[SD, 2] = InlineArray[SD, 2](-1.0, 1.0),
    ):
        self.id = id
        self.n_in = n_in
        self.n_out = n_out
        self.neurons = List[NT](capacity=self.n_out)

        self.do_phi_caching = PHI_CACHING and NT.phi_caching_capable()
        for i in range(self.n_out):
            self.neurons.append(
                NT(
                    NeuronID(self.id, i),
                    self.n_in,
                    num_trainable_edge_params,
                    weights_range,
                    i == 0
                    or not self.do_phi_caching,  # only first neuron in layer calculates
                )
            )

        self.xin = MV(n_in)  # input, shape (n_in,)
        self.xout = MV(n_out)  # output, shape (n_out,)
        self.dloss_dxin = MV(n_in)  # d loss / d xin, shape (n_in,)

        self.phi_mat = MM(num_trainable_edge_params, self.n_in)
        self.phi_der_mat = MM(num_trainable_edge_params, self.n_in)
        self.zero_grad()

    fn __call__(mut self, x: MV) -> MV:
        # forward pass
        self.xin = x
        for i in range(self.n_out):
            self.xout[i] = self.neurons[i](
                self.xin, self.phi_mat, self.phi_der_mat
            )
        return self.xout

    fn zero_grad(
        mut self, which: List[String] = List[String]("xin", "weights", "bias")
    ):
        for w in which:
            if w == "xin":  # reset layer's d loss / d xin
                self.dloss_dxin.zero()

            elif w == "weights":  # reset d loss / dw to zero for every neuron
                for nn in self.neurons:
                    nn.zero_dloss_dw()

            elif w == "bias":  # reset d loss / db to zero for every neuron
                for i in range(len(self.neurons)):
                    self.neurons[i].zero_dloss_dbias()

    fn update_grad(mut self, dloss_dxout: MV) -> MV:
        # update gradients by chain rule

        for i in range(dloss_dxout.size):
            # update layer's d loss / d xin via chain rule
            self.dloss_dxin += self.neurons[i].get_dxout_dxin() * dloss_dxout[i]

            # update neuron's d loss / dw and d loss / d bias
            self.neurons[i].update_dloss_dw_dbias(dloss_dxout[i])

        # print(self.dloss_dxin)
        return self.dloss_dxin
