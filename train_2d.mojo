
from python import Python
from sys.terminate import exit 

from kamo import MN,MM,MV,SD,SD2
from kamo.feed_forward_network import FeedForward
from kamo.func import relu
from kamo.func.edge import BSplineSilu,ChebyshevPolynomial,GaussianRBF
from kamo.func.loss import SquaredLoss
from kamo.neuron import NeuronType,NeuronKAN,NeuronNN

fn main() raises:

    alias SEED = 109

    # KAN compile time settings
    alias NUM_TRAINABLE_EDGE_PARAMS_BSPLINE = 7
    alias NUM_TRAINABLE_EDGE_PARAMS_CHEBYSHEV = 3
    alias NUM_TRAINABLE_EDGE_PARAMS_GAUSSIAN= 4
    alias X_BOUNDS = SD2(-1,1)
    alias PHI_CACHING = True
   
    # NN compile time setting
    alias ACTIVATION_NN = relu

    ## training  parameter

    var n_iter_train = 5000
    var loss_tolerance = 0.05
    var learning_rate_kan_bspline = 0.01
    var learning_rate_kan_cheby = 0.005
    var learning_rate_kan_gaussian = 0.0075
    var learning_rate_nn = 0.005
    
    # Training data

    var x1_num = 8
    var x2_num = 10

    fn fun2d(x1:SD,x2:SD)->SD:
        return x1 * MN.pow(x2, 0.5)
  
    var x_train = MM(x1_num*x2_num,2) # 2 input values
    var y_train = MM(x1_num*x2_num,1) # 1 output value

    var x1 = MN.linspace(0, 0.8, x1_num)
    var x2 = MN.linspace(0, 1, x2_num)

    var pos = 0
    for j in range(x2_num):
        for i in range(x1_num):
            x_train.insert_row(pos,MV(2,x1[i],x2[j]))
            y_train.insert_row(pos,MV(1,fun2d(x1[i],x2[j])))
            pos+=1

    # KAN 2D

    print("KAN training (BSpline edges)")

    var kan1 = FeedForward
        [
            NeuronKAN[BSplineSilu[3],
            X_BOUNDS
        ],
            SquaredLoss,
            PHI_CACHING
        ]
        (
            List[Int](2,2, 1),
            num_trainable_edge_params=NUM_TRAINABLE_EDGE_PARAMS_BSPLINE,
            learning_rate=learning_rate_kan_bspline,
            weights_range=SD2(-.1, .1),
            seed_val=SEED
        )
                       
    kan1.train(x_train, 
              y_train, 
              n_iter_max=n_iter_train, 
              loss_tolerance=loss_tolerance)

    
    print("\nKAN training (Chebyshev Polynomal edges)")

    var kan2 = FeedForward
        [
            NeuronKAN[ChebyshevPolynomial,
            X_BOUNDS
        ],
            SquaredLoss,
            PHI_CACHING
        ]
        (
            List[Int](2,2, 1),
            num_trainable_edge_params=NUM_TRAINABLE_EDGE_PARAMS_CHEBYSHEV,
            learning_rate=learning_rate_kan_cheby,
            weights_range=SD2(-.1, .1),
            seed_val=SEED
        )
                       
    kan2.train(x_train, 
              y_train, 
              n_iter_max=n_iter_train, 
              loss_tolerance=loss_tolerance)

    print("\nKAN training (Gaussian RBF edges)")

    var kan3 = FeedForward
        [
            NeuronKAN[GaussianRBF,
            X_BOUNDS
        ],
            SquaredLoss,
            PHI_CACHING
        ]
        (
            List[Int](2,2, 1),
            num_trainable_edge_params=NUM_TRAINABLE_EDGE_PARAMS_GAUSSIAN,
            learning_rate=learning_rate_kan_gaussian,
            weights_range=SD2(-.1, .1),
            seed_val=SEED
        )
                       
    kan3.train(x_train, 
              y_train, 
              n_iter_max=n_iter_train, 
              loss_tolerance=loss_tolerance)



    # MLP 2D
    
    print("\nMLP training")

    var mlp = FeedForward
        [
            NeuronNN[relu],
            SquaredLoss
        ]
        (
            List[Int](2, 6, 1), 
            num_trainable_edge_params=1, # layer size
            learning_rate=learning_rate_nn,
            weights_range=SD2(-.1, .1),
            seed_val=SEED
        )
    
    mlp.train(x_train, 
             y_train, 
             n_iter_max=n_iter_train, 
             loss_tolerance=loss_tolerance)
    

    ## plots

    var m1:MM
    var m2:MM

    m1,m2 = MN.meshgrid(x1,x2)
    var ym = MM(x2.size,x1.size)
    var kan1m = MM(x2.size,x1.size)
    var kan2m = MM(x2.size,x1.size)
    var kan3m = MM(x2.size,x1.size)
    var mlpm = MM(x2.size,x1.size)
    
    for i in range(m1.rows):
        for j in range(m2.cols):
            ym[i,j] = y_train[i*m1.cols+j]
            kan1m[i,j] = kan1(MV(2,m1[i,j],m2[i,j]))[0]
            kan2m[i,j] = kan2(MV(2,m1[i,j],m2[i,j]))[0]
            kan3m[i,j] = kan3(MV(2,m1[i,j],m2[i,j]))[0]
            mlpm[i,j] = mlp(MV(2,m1[i,j],m2[i,j]))[0]

    var plt = Python.import_module("matplotlib.pyplot")

    var fig = plt.figure(figsize=(14,9))

    var ax1 = fig.add_subplot(321)  # First row, middle column
    var ax2 = fig.add_subplot(323)  # Second row, first column
    var ax3 = fig.add_subplot(324)  # Second row, second column
    var ax4 = fig.add_subplot(325)  # Third row, first column
    var ax5 = fig.add_subplot(326)  # Third row, second column

    
    var vmin:SD 
    var vmax:SD
    vmin, vmax = MN.minmax(y_train)

    var im1 = ax1.pcolor(m1.to_numpy(), m2.to_numpy(), ym.to_numpy(), vmin=vmin, vmax=vmax)
    fig.colorbar(im1, ax=ax1)
    ax1.set_title('Training data',fontsize=20)
    
    var im2 = ax2.pcolor(m1.to_numpy(), m2.to_numpy(), kan1m.to_numpy(), vmin=vmin, vmax=vmax)
    fig.colorbar(im2, ax=ax2)
    ax2.set_title('KAN B-Spline',fontsize=20)
    
    var im3 = ax3.pcolor(m1.to_numpy(), m2.to_numpy(), kan2m.to_numpy(), vmin=vmin, vmax=vmax)
    fig.colorbar(im3, ax=ax3)
    ax3.set_title('KAN Chebyshev',fontsize=20)

    var im4 = ax4.pcolor(m1.to_numpy(), m2.to_numpy(), kan3m.to_numpy(), vmin=vmin, vmax=vmax)
    fig.colorbar(im4, ax=ax4)
    ax4.set_title('KAN Gaussian RBF',fontsize=20)

    var im5 = ax5.pcolor(m1.to_numpy(), m2.to_numpy(), mlpm.to_numpy(), vmin=vmin, vmax=vmax)
    fig.colorbar(im5, ax=ax5)
    ax5.set_title('MLP',fontsize=20)
#
    fig.tight_layout()

    plt.savefig("imgs/train_2d.png")
    plt.show()
