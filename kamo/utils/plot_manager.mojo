from python import Python
from os.system import remove

from kamo import dtype,simd_width
from kamo.libs.monum import MoVector

alias SD = Scalar[dtype]
alias MV = MoVector[dtype,simd_width]

struct PlotManager:

    var plt:PythonObject

    fn __init__(inout self) raises:
        self.plt = Python.import_module("matplotlib.pyplot")

    fn save_prediction_graph(self,
        x_train:MV, 
        y_train:MV, 
        y_pred:MV, 
        title:String,
        name_train:String,
        name_pred:String,
        path:String,
        empty_folder:Bool=True) raises:
        
        self.plt.figure(figsize=(10, 6))
        self.plt.plot(x_train.to_numpy(), y_train.to_numpy(), label=name_train, color='blue')
        self.plt.plot(x_train.to_numpy(), y_pred.to_numpy(), label=name_pred, color='red')
        self.plt.title(title)
        self.plt.xlabel('x')
        self.plt.ylabel('y')
        self.plt.legend()
        self.plt.grid(True)
        self.plt.savefig(path)
        self.plt.close()