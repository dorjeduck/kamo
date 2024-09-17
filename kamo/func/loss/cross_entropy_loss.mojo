from math import log

from kamo import MN,MM,MV,SD,SD2

struct CrossEntropyLoss(LossFunction):
    
    var n_in: Int
    var y: MV
    var dloss_dy: MV
    var loss: SD
    var y_train: MV

    fn __init__(inout self, n_in:Int):
        self.n_in = n_in

        self.y = MV(0)
        self.dloss_dy = MV(0)
        self.loss = 0
        self.y_train = MV(0)
    
    fn __call__(inout self, y: MV, y_train: MV) -> SD:
        # y: output of network
        # y_train: ground truth
        self.y = y
        self.y_train = y_train
        self.calc_loss()
        self.calc_dloss_dy()
        return self.loss

    fn get_loss(self)->SD:
        return self.loss
    
    fn get_dloss_dy(self)->MV:
        return self.dloss_dy

    fn calc_loss(inout self):
        self.loss = - log(MN.exp(self.y[int(self.y_train[0])]) / MN.sum(MN.exp(self.y)))

    fn calc_dloss_dy(inout self):
        self.dloss_dy = MN.exp(self.y) / MN.sum(MN.exp(self.y))    
        for i in range(self.y_train.size):
            self.dloss_dy[int(self.y_train[i])] -= 1
        