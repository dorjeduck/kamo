from random.random import seed
from time import now,sleep

from kamo import dtype,simd_width
from kamo.libs.monum import MoNum,MoVector,PI
from kamo.nn import Edge
from kamo.func import SquaredLoss
from kamo.func.edge import BSplineSilu

from kamo.libs.mopro import progress_bar,BarSettings

from kamo.utils import PlotManager

alias SD = Scalar[dtype]
alias MV = MoVector[dtype,simd_width]
alias MN = MoNum[dtype,simd_width]


fn main() raises:

    #seed(now())

    alias img_output = False

    var num_trainable_params = 11
    var epochs=20000
    var name_pred = "BSplineSilu"
    var name_train = "sin()"
    
    var learning_rate=0.1
    var x_bounds = List[SD](0,4*PI)

    # Edge

    var edge = Edge[BSplineSilu[3]](x_bounds,num_trainable_params)

    # Training data
    var x_train = MN.linspace(x_bounds[0], x_bounds[1], 101)
    var y_train = MN.sin(x_train)


    # Training
    
    var y_pred = edge(x_train)

     # Image output 

    var pm = PlotManager()
    
    @parameter
    fn save_image(step:Int) raises:
        if img_output:
           
            pm.save_prediction_graph(
                        x_train,
                        y_train,
                        y_pred,
                        "Epoch " + str(step),
                        name_train,
                        name_pred,
                        "imgs/BSplineSilu-" + str(step) + ".png"
                    )
         
            
    save_image(0)

    var start = now()

    for step in range(epochs+1):
        #forward pass
        var y_pred = edge(x_train)
       
        #calculate loss
        var loss = SquaredLoss.loss(y_pred,y_train)

        #backward pass
        var dloss_dy = SquaredLoss.dloss_dy(y_pred,y_train)
        var gradients = edge.calc_gradients(x_train,dloss_dy)
       
        edge.update_weights(-learning_rate * gradients)

        if (step+1)%100 == 0:
            print("Epoch: " + str(step+1) + ", loss:" + str(loss))
            
            save_image(step+1)
    
    var elapased = (now()-start)/1e9

    print("Training time:",elapased,"sec")