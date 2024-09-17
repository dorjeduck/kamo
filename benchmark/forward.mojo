from kamo import MN,MM,MV,SD,SD2
from kamo.feed_forward_network import FeedForward
from kamo.func.edge import GaussianRBF
from kamo.func.loss import CrossEntropyLoss
from kamo.neuron import NeuronType,NeuronKAN



fn main() raises:

    alias NUM_TRAINABLE_EDGE_PARAMS_BSPLINE = 8
    alias PHI_CACHING = True
    alias X_BOUNDS = SD2(-1,1)
    alias SEED = 444

    ## training  parameter
    var n_max_iter_train = 5000
    var loss_tolerance = 0.005
    var learning_rate_kan = 0.01
    
    var num_pix = 28*28
    var num_rows = 60

    var path = 'data/mnist_csv/mnist_train.csv'

    var f = open(path,"r")
    var lines = f.read().split("\n")  
   
    var x_train = MM(num_rows,num_pix)
    var y_train = MM(num_rows,1)
    
    print("Loading data")

    for i in range(num_rows):
        var data = lines[i+1].split(",")
        y_train[i,0] = int(data[0])
        for j in range(num_pix):
            x_train[i,j] = int(data[j+1])
         
    var kan1 = FeedForward
        [
            NeuronKAN[GaussianRBF,X_BOUNDS],
            CrossEntropyLoss,
            PHI_CACHING
        ]
        (
            List[Int](num_pix, 64, 10),
            num_trainable_edge_params=NUM_TRAINABLE_EDGE_PARAMS_BSPLINE,
            learning_rate=learning_rate_kan,
            weights_range=SD2(-1, 1),
            seed_val=SEED
        )

        