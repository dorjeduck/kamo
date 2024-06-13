
from python import Python

from kamo import MN,MM,SD2
from kamo.func.edge import ChebyshevPolynomial

fn main() raises:

    alias NUM_PLOT_DATA = 200
    alias X_BOUNDS = SD2(-1,1)
    alias NUM_TRAINABLE_EDGE_PARAMS = 8
   
    var func = ChebyshevPolynomial(NUM_TRAINABLE_EDGE_PARAMS,X_BOUNDS)

    ## plot

    var x_plot = MN.linspace(X_BOUNDS[0], X_BOUNDS[1],NUM_PLOT_DATA) 

    var phi = MM(NUM_TRAINABLE_EDGE_PARAMS,NUM_PLOT_DATA)
    func.calc_phi_mat(phi,x_plot)

    var plt = Python.import_module("matplotlib.pyplot")

    var fig = plt.figure(figsize=(12,8))
    var ax = fig.add_subplot(111)
    
    var colors = InlineArray[String,10](
    'darkred',  
    'navy', 
    'firebrick', 
    'darkmagenta', 
    'darkorange', 
    'indigo', 
    'saddlebrown',
    'teal',
    )

    var x_plot_numpy = x_plot.to_numpy()

    for i in range(NUM_TRAINABLE_EDGE_PARAMS):
        ax.plot(x_plot_numpy, phi.get_row(i).to_numpy(), color=colors[i%8])
    ax.set_title("Chebyshev Polynomials basis functions",fontsize=24)
    ax.grid()
    fig.tight_layout()

    plt.savefig("imgs/chebyshev_basis.png")
    plt.show()
    