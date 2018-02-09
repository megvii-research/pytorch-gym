from basic_utils.utils import *


class ConcatFixedStd(nn.Module):
    """
    Add a fixed standard err to the input vector.
    """
    def __init__(self, ishp):
        super(ConcatFixedStd, self).__init__()
        self.log_var = nn.Parameter(torch.zeros(1, ishp) - 1.0)

    def forward(self, x):
        Mean = x
        Std = torch.exp(self.log_var * 0.5) * Variable(torch.ones(x.size()))
        return torch.cat((Mean, Std), dim=1)


class Flatten(nn.Module):
    """
    The flatten module.
    """
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size()[0], -1)


class Add_One(nn.Module):
    """
    A module whose function is to add one to the input.
    """
    def __init__(self):
        super(Add_One, self).__init__()

    def forward(self, x):
        return x+1


class Softplus(nn.Module):
    """
    The softplus module.
    """
    def __init__(self):
        super(Softplus, self).__init__()

    def forward(self, x):
        return (1 + x.exp()).log()


def get_layer(des, inshp):
    """
    Get a torch layer according to the description.

    Args:
        des: the description of the layer.
        inshp: input shape

    Return:
        layer: the corresponding torch network
        inshp: the output shape
    """
    if des['kind'] == 'conv':
        return nn.Conv2d(in_channels=inshp, out_channels=des["filters"], kernel_size=des["ker_size"],
                         stride=des["stride"]), des["filters"]
    if des['kind'] == 'flatten':
        return Flatten(), 3136
    if des["kind"] == 'dense':
        return nn.Linear(in_features=inshp, out_features=des["units"]), des["units"]
    if des['kind'] == 'ReLU':
        return nn.ReLU(), inshp
    if des['kind'] == 'Tanh':
        return nn.Tanh(), inshp
    if des['kind'] == 'Dropout':
        return nn.Dropout(p=des['p']), inshp
