import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class SimpleNeuralNetwork(nn.Module) :

    def __init__(self, hidden_sizes, out_size) :
        super(SimpleNeuralNetwork, self).__init__()

        self.hidden = nn.ModuleList()
        for k in range(len(hidden_sizes)-1):
            self.hidden.append(nn.Linear(hidden_sizes[k], hidden_sizes[k+1]))

        self.out = nn.Linear(hidden_sizes[-1], out_size)

    def forward(self, x) :
        # Feedforward
        for layer in self.hidden:
            x = F.leaky_relu(layer(x))
        output= F.softmax(self.out(x), dim=1)

        return output

    def display(self) :
        print("\n##### Model infos #####")
        params = list(self.parameters())
        print("Number of parameters : ", len(params))
        print(self, "\n")
