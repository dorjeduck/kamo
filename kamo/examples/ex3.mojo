from random.random import seed
from time import now,sleep

from kamo import dtype,simd_width
from kamo.libs.monum import MoNum,MoVector,PI
from kamo.nn import Edge
from kamo.func import SquaredLoss
from kamo.func.edge import BSpline

from kamo.libs.mopro import progress_bar,BarSettings

from kamo.utils import PlotManager

alias SD = Scalar[dtype]
alias MV = MoVector[dtype,simd_width]
alias MN = MoNum[dtype,simd_width]


   


fn main() raises:

    seed(now())

    var epochs=30000

    var name_pred = "BSpline"
    var name_train = "sin()"
    
    var learning_rate=0.1
    var x_bounds = List[SD](0,4*PI)

    # Training data
    var x_train = MN.linspace(x_bounds[0], x_bounds[1], 101)
    var y_train = MN.sin(x_train)

    # Edge function
    var n_func = 11
    
    var edge = Edge[BSpline[3]](x_bounds,n_func)

    # Training

    var pm = PlotManager()

    var y_pred = edge(x_train)

    pm.save_prediction_graph(
                x_train,
                y_train,
                y_pred,
                "Epoch " + str(0),
                name_train,
                name_pred,
                "imgs/bspline-" + str(0) + ".png"
            )
    
    for step in range(epochs+1):
        #forward pass
        var y_pred = edge(x_train)
       
        #calculate loss
        var loss = SquaredLoss.loss(y_pred,y_train)

        #backward pass
        var dloss_dy = SquaredLoss.dloss_dy(y_pred,y_train)
        var gradients = edge.get_gradients(x_train,dloss_dy)
       
        edge.update_weights(-learning_rate * gradients)

        if (step+1)%100 == 0:
            print("Epoch: " + str(step+1) + ", loss:" + str(loss))
            pm.save_prediction_graph(
                x_train,
                y_train,
                y_pred,
                "Epoch " + str(step+1),
                name_train,
                name_pred,
                "imgs/bspline-" + str(step+1) + ".png"
            )
            

   

    
    




   


