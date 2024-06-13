from random.random import seed
from time import now 

from kamo import MN,MM,MV,SD,SD2
from kamo.feed_forward_network import FullyConnectedLayer
from kamo.func import relu, tanh_act, ACF
from kamo.func.edge import EdgeFunc, BSplineSilu
from kamo.func.loss import LossFunction, SquaredLoss
from kamo.libs.mopro import progress_bar, BarSettings
from kamo.neuron import NeuronType
from kamo.utils import str_maxlen

struct FeedForward[NT: NeuronType, LF: LossFunction,PHI_CACHING:Bool=True]:
    var layers: List[FullyConnectedLayer[NT,PHI_CACHING]]
    var layer_len: List[Int]
    var learning_rate: SD
    var n_layers: Int
    var loss_hist: MV

    var loss: LF

    fn __init__(inout self, 
        layer_len: List[Int], 
        num_trainable_edge_params:Int,
        learning_rate: SD = 0.01,
        weights_range:SD2 = SD2(-1.0, 1.0),
        seed_val:Int=-1
        ):
        
        self.layer_len = layer_len
        self.learning_rate = learning_rate
        self.n_layers = self.layer_len.size - 1
        self.layers = List[FullyConnectedLayer[NT,PHI_CACHING]](capacity=self.n_layers)

        for ii in range(self.n_layers):
            self.layers.append(
                FullyConnectedLayer[NT,PHI_CACHING](ii, layer_len[ii], layer_len[ii + 1],
                num_trainable_edge_params,
                weights_range)
            )

        self.loss = LF(self.layer_len[-1])
        self.loss_hist = MV(0)

        if seed_val == -1:
            seed(now())
        else:
            seed(seed_val)
        
    fn __call__(inout self, x: MV) -> MV:
        # forward pass
        var x_in_out = x

        for l in range(self.n_layers):
            x_in_out = self.layers[l](x_in_out)

        return x_in_out

    fn __call__(inout self, x_batch: MM) -> MM:
        # batch forward pass

        var res = MM(x_batch.rows,self.layer_len[-1])
        
        for i in range(x_batch.rows):
            res.insert_row(i,self(x_batch.get_row(i)))

        return res


    fn backprop(inout self):
        # gradient backpropagation
        var delta = self.layers[-1].update_grad(self.loss.get_dloss_dy())

        for ll in range(self.n_layers - 2, -1, -1):
            delta = self.layers[ll].update_grad(delta)

    fn gradient_descent_par(inout self):
        # update parameters via gradient descent
        for ll in self.layers:
            for nn in ll[].neurons:
                nn[].gradient_descent(self.learning_rate)

    fn train(
        inout self,
        x_train: MM,
        y_train: MM,
        n_iter_max: Int = 10000,
        loss_tolerance: SD = 0.01,
    ):
        self.loss_hist = MV(n_iter_max)

        @parameter
        fn _iter(it: Int, inout bs: BarSettings) -> Bool:
            var loss: SD = 0.0  # reset loss

            for ii in range(x_train.rows):
                # forward pass
                var x_out = self(x_train.get_row(ii))

                # accumulate loss
                loss += self.loss(x_out, y_train.get_row(ii))

                # zero 
                for layer in self.layers:
                    layer[].zero_grad(which=List[String]("xin"))

                # backward propagation
                self.backprop()

            self.loss_hist[it] = loss

            if loss < loss_tolerance:
                # print(loss, "Convergence has been attained!")
                bs.postfix = (
                    "loss: "
                    + str_maxlen(str(loss), 6)
                    + ", Convergence!"
                )
                return False
            if it % 10 == 0:
                bs.postfix = "loss: " + str_maxlen(str(loss), 6)
                
            # update parameters

            self.gradient_descent_par()  

            for layer in self.layers:
                layer[].zero_grad(
                    which=List[String]("weights", "bias")
                )  # reset gradient wrt par to zero

            return True

        progress_bar[_iter](n_iter_max,
        bar_size=10,
        bar_fill = "🔥",
        bar_empty = "  "
        )

