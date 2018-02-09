from basic_utils.utils import *


class ValueFunction:
    def __init__(self, net, optimizer):
        self.net = net
        self.optimizer = optimizer

    def predict(self, ob):
        observations = turn_into_cuda(np_to_var(np.array(ob)))
        return self.net(observations).data.cpu().numpy()

    def fit(self, path):
        stat, info = self.optimizer(path)
        return 'v', stat, info

    def save_model(self, name):
        torch.save(self.net, name + "_baseline.pkl")

    def load_model(self, name):
        net = torch.load(name + "_baseline.pkl")
        self.net.load_state_dict(net.state_dict())
