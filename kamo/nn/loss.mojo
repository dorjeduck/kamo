from kamo import dtype,simd_width
from kamo.libs.monum import MoVector,MoNum

alias SD = Scalar[dtype]
alias MV = MoVector[dtype,simd_width]
alias MN = MoNum[dtype,simd_width]

#alias LOSS_FUNC = fn (MV,MV,Bool) -> SD:
#
#fn mse(y:MV,y_train:MV,grad:Bool=False) -> MV:
#    if not grad:
#        return MN.pow(y-y_train)
#    else:
#        return 2 * (y - y_train) / SD(len(y))



trait LossFunction:

    @staticmethod
    fn loss(y:MV,y_train:MV) -> SD: pass

    @staticmethod
    fn dloss_dy(y:MV,y_train:MV) -> MV: pass
        

struct SquaredLoss(LossFunction):
    @staticmethod
    fn loss(y:MV,y_train:MV) -> SD:
        return MN.mean(y-y_train)

    @staticmethod
    fn dloss_dy(y:MV,y_train:MV) -> MV:
        return 2 * (y - y_train) /SD(len(y))


'''
struct CrossEntropyLoss(LossFunction):
    
    var y:MoVector[dtype]
    var y_train:MoVector[dtype]
    var dloss_dy:MoVector[dtype]
    var loss:Scalar[dtype]

    fn __init__(self, n_in:Int):
        n_in = n_in
    
    fn __call__(self, y:MoVector[dtype],y_train: MoVector[dtype]) -> Scalar[dtype]:
        # y: output of network
        # y_train: ground truth
        y = y
        y_train = y_train
        calc_loss()
        calc_dloss_dy()
        return loss

    fn calc_loss(self,) :
        # compute loss l(xin, y)
        loss = - log(exp(y[y_train[0]]) / sum(exp(y)))

    fn calc_dloss_dy(self, inout dloss_dy,inout y,inout y_train):
        # compute gradient of loss wrt xin
        dloss_dy = exp(y) / sum(exp(y))
        dloss_dy[y_train] -= 1
'''
