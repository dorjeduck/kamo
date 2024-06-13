from utils.static_tuple import InlineArray

from kamo import MN,MM,MV,SD,SD2
from kamo.func.edge import EdgeFunc, BSplineSilu
from kamo.func import tanh_act, ACF
from kamo.neuron import NeuronType

@value
struct NeuronKAN[EF: EdgeFunc = BSplineSilu[3],x_bounds:SD2= SD2(-1,1)](
    NeuronType
):
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

    var edge_func: EF  # edge functions 
    var activation: ACF  # neuron activation function

    var phi_mat: MM
    var phi_der_mat: MM

    var unit_vec:MV

    var id:NeuronID

    var calc_phi_mat:Bool

    fn __init__(
        inout self,
        id:NeuronID,
        n_in: Int,
        num_trainable_edge_params:Int,
        weights_range: SD2,
        calc_phi_mat:Bool = True
    ):
        self.id = id
        self.n_in = n_in
        self.weights = MM.rand(
            num_trainable_edge_params,
            self.n_in,
            weights_range[0],
            weights_range[1],
        )

        self.edge_func = EF(num_trainable_edge_params, x_bounds)

        self.xin = MV(self.n_in)
        self.xmid = MV(self.n_in)
        
        self.xout = 0
        self.bias = 0

        self.dxout_dxmid = MV(self.n_in)
        self.dxout_dbias = 0
        self.dxmid_dw = MM(self.n_in, num_trainable_edge_params)
        self.dxmid_dxin = MV(self.n_in)
        self.dxout_dxin = MV(self.n_in)
        self.dxout_dw = MM(self.n_in, num_trainable_edge_params)
        self.dloss_dw = MM(self.n_in, num_trainable_edge_params)
        self.dloss_dbias = 0

        self.activation = tanh_act  # normalization (for splines...)
       
        self.phi_mat = MM(num_trainable_edge_params,self.n_in)
        self.phi_der_mat = MM(num_trainable_edge_params,self.n_in)

        self.unit_vec = MV(self.n_in,1.0)

        self.calc_phi_mat = calc_phi_mat

    ## common to all neuron types

    fn __call__(inout self, x: MV) -> SD:

        MN.inplace_copy(self.xin,x)

        # forward pass: compute neuron's output
        self.calc_xmid()
        self.calc_xout()

        # compute internal derivatives
        self.calc_dxout_dxmid()
        self.calc_dxout_dbias()
        self.calc_dxmid_dw()
        self.calc_dxmid_dxin()

        # compute external derivativesdxout_dxin
        self.calc_dxout_dxin()
        self.calc_dxout_dw()

        # return neuron output
        return self.xout

    fn calc_dxout_dxin(inout self):
        MN.inplace_copy(self.dxout_dxin,self.dxout_dxmid * self.dxmid_dxin)

    fn get_dxout_dxin(self) -> MV:
        return self.dxout_dxin

    fn calc_dxout_dw(inout self):
        MN.inplace_copy(self.dxout_dw ,MN.diag(self.dxout_dxmid) @ self.dxmid_dw)

    fn update_dloss_dw_dbias(inout self, dloss_dxout: SD):
        self.dloss_dw += self.dxout_dw * dloss_dxout
        self.dloss_dbias += self.dxout_dbias * dloss_dxout

    fn gradient_descent(inout self, learning_rate: SD):
        self.weights -= learning_rate * self.dloss_dw
        self.bias -= learning_rate * self.dloss_dbias

    fn zero_dloss_dw(inout self):
        self.dloss_dw.zero()

    fn zero_dloss_dbias(inout self):
        self.dloss_dbias = 0.0

    ## kan specific

    fn calc_xmid(inout self):
        if self.calc_phi_mat:
            self.edge_func.calc_phi_mat(self.phi_mat, self.xin)
        MN.inplace_copy(self.xmid,MN.sum(self.weights * self.phi_mat, axis=0))

    fn calc_xout(inout self):
        self.xout = self.activation(MN.sum(self.xmid))

    fn calc_dxout_dxmid(inout self):
        MN.inplace_copy(self.dxout_dxmid,self.activation(
            Scalar[dtype](MN.sum(self.xmid)), True
        ) * self.unit_vec)

    fn calc_dxmid_dw(inout self):
        MN.inplace_copy(self.dxmid_dw,self.phi_mat)
       
    fn calc_dxmid_dxin(inout self):
        if self.calc_phi_mat:
            self.edge_func.calc_phi_mat(self.phi_der_mat, self.xin, True)
        MN.inplace_copy(self.dxmid_dxin,MN.sum(self.weights * self.phi_der_mat, axis=0))

    fn calc_dxout_dbias(inout self):
        # no bias in KAN!
        self.dxout_dbias = 0

    fn get_phi_mat(self)->MM:
        return self.phi_mat
    
    fn get_phi_der_mat(self)->MM:
        return self.phi_der_mat

    fn set_phi_mat(inout self,pm:MM):
        MN.inplace_copy(self.phi_mat,pm)
    
    fn set_phi_der_mat(inout self,pdm:MM):
        MN.inplace_copy(self.phi_der_mat,pdm)

    @staticmethod
    fn phi_caching_capable()->Bool:
        return True