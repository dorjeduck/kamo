from python import Python

from kamo import MN,MM,MV,SD,SD2
from kamo.feed_forward_network import FeedForward
from kamo.func import relu
from kamo.func.loss import SquaredLoss
from kamo.func.edge import BSplineSilu,ChebyshevPolynomial,GaussianRBF
from kamo.neuron import NeuronType,NeuronKAN,NeuronNN

fn main() raises:

    alias SEED = 444
    alias NUM_TRAINING_DATA = 30

    # KAN compile time settings
    alias NUM_TRAINABLE_EDGE_PARAMS_BSPLINE = 7
    alias NUM_TRAINABLE_EDGE_PARAMS_CHEBYSHEV = 3
    alias NUM_TRAINABLE_EDGE_PARAMS_GAUSSIAN = 4
    alias X_BOUNDS = SD2(-1,1)
    alias PHI_CACHING = True
   
    # NN compile time setting
    alias ACTIVATION_NN = relu

    ## training  parameter
    var n_max_iter_train = 5000
    var loss_tolerance:SD = 0.005
    var learning_rate_kan:SD = 0.01
    var learning_rate_nn:SD = 0.005
   
    # Training data

    var x_train = MN.linspace(X_BOUNDS[0], X_BOUNDS[1],NUM_TRAINING_DATA, 1)   
    var y_train = .5 * MN.sin(4 * x_train) * MN.exp(-(x_train+1.)) + .5  
         
    # KAN 1D

    print("KAN training (BSpline edges)")

    var kan1 = FeedForward
        [
            NeuronKAN[BSplineSilu[3],X_BOUNDS],
            SquaredLoss,
            PHI_CACHING
        ]
        (
            List[Int](1,2,2,1),
            num_trainable_edge_params=NUM_TRAINABLE_EDGE_PARAMS_BSPLINE,
            learning_rate=learning_rate_kan,
            weights_range=SD2(-1, 1),
            seed_val=SEED
        )
                       
    kan1.train(x_train, 
              y_train, 
              n_iter_max=n_max_iter_train, 
              loss_tolerance=loss_tolerance)

    print("\nKAN training (Chebyshev Polynominal edges)")

    var kan2 = FeedForward
        [
            NeuronKAN[ChebyshevPolynomial,X_BOUNDS],
            SquaredLoss,
            PHI_CACHING
        ]
        (
            List[Int](1,2,2,1),
            num_trainable_edge_params=NUM_TRAINABLE_EDGE_PARAMS_CHEBYSHEV,
            learning_rate=learning_rate_kan,
            weights_range=SD2(-1, 1),
            seed_val=SEED
        )
             
    kan2.train(x_train, 
              y_train, 
              n_iter_max=n_max_iter_train, 
              loss_tolerance=loss_tolerance)

    
    print("\nKAN training (Gaussian RBF edges)")

    var kan3 = FeedForward
        [
            NeuronKAN[GaussianRBF,X_BOUNDS],
            SquaredLoss,
            PHI_CACHING
        ]
        (
            List[Int](1,2,2,1),
            num_trainable_edge_params=NUM_TRAINABLE_EDGE_PARAMS_GAUSSIAN,
            learning_rate=learning_rate_kan,
            weights_range=SD2(-1, 1),
            seed_val=SEED
        )
             
    kan3.train(x_train, 
              y_train, 
              n_iter_max=n_max_iter_train, 
              loss_tolerance=loss_tolerance)

    

    # MLP 1D
    
    print("\nMLP training")

    var mlp = FeedForward
        [
            NeuronNN[relu],
            SquaredLoss
        ]
        (
            List[Int](1, 13, 1), 
            num_trainable_edge_params=1, # layer size
            learning_rate=learning_rate_nn,
            weights_range=SD2(-.5, .5),
            seed_val=SEED
        )
    
    mlp.train(x_train, 
             y_train, 
             n_iter_max=n_max_iter_train, 
             loss_tolerance=loss_tolerance)

    
    ## plot

    var num_points=1000

    var x_plot = MN.linspace(x_train[0,0], x_train[x_train.rows-1,0],num_points,1)  
    
    var y_plot_kan1 = kan1(x_plot).get_col(0)
    var y_plot_kan2 = kan2(x_plot).get_col(0)
    var y_plot_kan3 = kan3(x_plot).get_col(0)
    
    var y_plot_mlp = mlp(x_plot).get_col(0)

    var plt = Python.import_module("matplotlib.pyplot")

    var fig = plt.figure(figsize=(10,7))
    var ax = fig.add_subplot(111)

    var x_numpy = x_plot.get_col(0).to_numpy()
    
    ax.plot(x_train.get_col(0).to_numpy(), y_train.get_col(0).to_numpy(), 'o', color="blue", label='Training Dataset')
    
    ax.plot(x_numpy, y_plot_kan1.to_numpy(), color='orange', label='KAN B-Spline')
    ax.plot(x_numpy, y_plot_kan2.to_numpy(), color='red', label='KAN Chebyshev')
    ax.plot(x_numpy, y_plot_kan3.to_numpy(), color='purple', label='KAN Gaussian RBF')
    ax.plot(x_numpy, y_plot_mlp.to_numpy(), color='green', label='MLP')
   
    #ax.set_xlabel('input feature', fontsize=13)
    #ax.set_title('Regression', fontsize=15)
    ax.legend()
    ax.grid()
    fig.tight_layout()

    plt.savefig("imgs/train_1d.png")
    plt.show()