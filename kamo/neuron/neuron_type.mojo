from kamo import MN,MM,MV,SD,SD2

trait NeuronType(CollectionElement):
    fn __init__(
        inout self, 
        id:NeuronID,
        n_in: Int, 
        num_trainable_edge_params:Int,
        weights_range:SD2,
        calc_phi_mat:Bool = True
    ):
        ...

    ## common

    fn __call__(inout self, xin:MV) -> SD:
        ...

    fn calc_dxout_dxin(inout self):
        ...

    fn calc_dxout_dw(inout self):
        ...

    fn update_dloss_dw_dbias(inout self, dloss_dxout: SD):
        ...

    fn gradient_descent(inout self, learning_rate: SD):
        ...

    fn get_dxout_dxin(self) -> MV:
        ...

    fn zero_dloss_dw(inout self):
        ...
    
    fn zero_dloss_dbias(inout self):
        ...
    
    # neuron type specific

    fn calc_xmid(inout self):
        ...

    fn calc_xout(inout self):
        ...

    fn calc_dxout_dxmid(inout self):
        ...

    fn calc_dxmid_dw(inout self):
        ...

    fn calc_dxmid_dxin(inout self):
        ...

    fn calc_dxout_dbias(inout self):
        ...

    fn get_phi_mat(self)->MM:
        ...
    
    fn get_phi_der_mat(self)->MM:
        ...

    fn set_phi_mat(inout self,pm:MM):
        ...
    
    fn set_phi_der_mat(inout self,pdm:MM):
        ...

    @staticmethod
    fn phi_caching_capable()->Bool:
        ...
