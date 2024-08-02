from torch import nn

class MillionsMOE(nn.Module):
    def __init__(self, args):
        super(MillionsMOE, self).__init__()

        self.args = args
