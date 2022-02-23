from turtle import forward
import torch
import torch.nn.functional as F
from torch.nn import Linear,Module

import pathlib
from torch.utils.tensorboard import SummaryWriter

class Network(Module):
    def __init__(self):
        super().__init__()

        self.fc_1 = Linear(10, 20)
        self.fc_2 = Linear(20, 30)
        self.fc_3 = Linear(30, 2)


    def forward(self, x):
        x = self.fc_1(x)
        # self.writer.add_histogram('1', x)
        x = self.fc_2(x)
        # self.writer.add_histogram('2', x)
        x = self.fc_3(x)
        # self.writer.add_histogram('3', x)

        x = F.relu(x)
        return x

if __name__ == "__main__":
    log_dir = pathlib.Path.cwd() / 'tensorboard_logs'
    writer = SummaryWriter(log_dir)
    x = torch.rand(1, 10)
    net = Network()

    def activation_hook(inst, inp, out):
        # inst: torch.nn.Module  The layer we want t attach the hook to.
        # inp: input to forward
        # out: output of forward

        print('Here')
        writer.add_histogram(repr(inst), out)

    net.fc_1.register_forward_hook(activation_hook)
    net.fc_2.register_forward_hook(activation_hook)
    net.fc_3.register_forward_hook(activation_hook)

    y = net(x)
    print(y)