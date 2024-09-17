from kamo import MN,MM,MV,SD,SD2

struct SquaredLoss(LossFunction):
    var n_in: Int
    var y: MV
    var dloss_dy: MV
    var loss: SD
    var y_train: MV

    fn __init__(inout self, n_in: Int):
        self.n_in = n_in

        self.y = MV(0)
        self.dloss_dy = MV(0)
        self.loss = 0
        self.y_train = MV(0)

    fn __call__(inout self, y: MV, y_train: MV) -> SD:
       
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
        self.loss = MN.mean(self.y - self.y_train)

    fn calc_dloss_dy(inout self):
        self.dloss_dy = 2 * (self.y - self.y_train) / SD(self.y.size)
