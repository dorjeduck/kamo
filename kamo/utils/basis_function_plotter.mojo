from python import Python

from kamo import MN, MM, SD2
from kamo.func.edge import EdgeFunc


struct BasisFunctionPlotter:
    var num_plot_data: Int
    var x_bounds: SD2
    var num_func: Int
    var colors: InlineArray[String,8]

    var plt: PythonObject

    fn __init__(
        inout self,
        num_func: Int,
        num_plot_data: Int,
        x_bounds: SD2 = SD2(-1, 1),
    ) raises:
        self.num_func = num_func
        self.num_plot_data = num_plot_data
        self.x_bounds = x_bounds

        self.colors = InlineArray[String, 8](
            "darkred",
            "navy",
            "firebrick",
            "darkmagenta",
            "darkorange",
            "indigo",
            "saddlebrown",
            "teal",
        )

        self.plt = Python.import_module("matplotlib.pyplot")

    fn plot[EF:EdgeFunc](inout self, inout func: EF, title: String, path: String,derivative:Bool=False) raises:
        ## plot

        var x_plot = MN.linspace(
            self.x_bounds[0], self.x_bounds[1], self.num_plot_data
        )

        var phi = MM(self.num_func, self.num_plot_data)
        func.calc_phi_mat(phi, x_plot,derivative)

        var fig = self.plt.figure(figsize=(12, 8))
        var ax = fig.add_subplot(111)

        var x_plot_numpy = x_plot.to_numpy()

        for i in range(self.num_func):
            ax.plot(
                x_plot_numpy, phi.get_row(i).to_numpy(), color=self.colors[i % 8]
            )
        ax.set_title(title, fontsize=24)
        ax.grid()
        fig.tight_layout()

        self.plt.savefig(path)
        self.plt.show()
