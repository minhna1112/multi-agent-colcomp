from re import I
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def init_hidden(layer):
    input_size = layer.weight.data.size()[0]
    lim = np.sqrt(1. / input_size)
    return (-lim, lim)

class ActorNN(nn.Module):
    """Actor (Policy) Model."""
    def __init__(self, input_size, output_size, seed):
        super(ActorNN, self).__init__()
        self.seed = torch.manual_seed(seed)
        hidden_nodes = 128
        self.num_layers = 3

        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_nodes)])
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(num_features=hidden_nodes)])
        for i in range(self.num_layers):
            self.hidden_layers.append(nn.Linear(hidden_nodes, hidden_nodes))
            self.bn_layers.append(nn.BatchNorm1d(num_features=hidden_nodes))
        self.final_layer = nn.Linear(hidden_nodes, output_size)
        self.init_params()

    def init_params(self):
        for i in range(len(self.hidden_layers)):
            nn.init.uniform_(self.hidden_layers[i].weight.data,*init_hidden(self.hidden_layers[i]))
        nn.init.uniform_(self.final_layer.weight.data, -3e-3, 3e-3)

    def forward(self, state):
        x = state
        for i in range(len(self.hidden_layers)):
            x = torch.relu(self.hidden_layers[i](x))
            x = self.bn_layers[i](x)
        x = torch.tanh(self.final_layer(x))
        return x