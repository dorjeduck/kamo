from kamo import MN,MM,MV,SD,SD2

trait EdgeFunc(CollectionElement):
    fn __init__(inout self, num_trainable_params: Int,x_bounds: SD2):
        pass

    fn calc_phi_mat(inout self,inout phi_mat:MM, x: MV, grad: Bool = False):
        pass
    