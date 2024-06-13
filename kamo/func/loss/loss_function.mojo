from kamo import MN,MM,MV,SD,SD2

trait LossFunction:
    fn __init__(inout self, n_in: Int):
        pass
    
    fn __call__(inout self, y:MV, y_train: MV) -> SD:
        pass

    fn get_loss(self)->SD:
        pass
        
    fn get_dloss_dy(self)->MV:
        pass
       
    fn calc_loss(inout self):
        pass

    fn calc_dloss_dy(inout self):
        pass
        