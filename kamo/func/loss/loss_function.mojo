from kamo import MN, MM, MV, SD, SD2


trait LossFunction:
    fn __init__(out self, n_in: Int):
        pass

    fn __call__(mut self, y: MV, y_train: MV) -> SD:
        pass

    fn get_loss(self) -> SD:
        pass

    fn get_dloss_dy(self) -> MV:
        pass

    fn calc_loss(mut self):
        pass

    fn calc_dloss_dy(mut self):
        pass
