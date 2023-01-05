from random import sample

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from sklearn.metrics import roc_auc_score

import torch
from torch.nn import ReLU
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import Sequential
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

class SimpleDenseNet(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, dropout=0.0):
        super(SimpleDenseNet, self).__init__()
        hidden_dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for (in_dim, out_dim) in zip(hidden_dims[:-1], hidden_dims[1:]):
            layers.append(Linear(in_dim, out_dim))
            layers.append(ReLU())
            if dropout > 0:
                layers.append(Dropout(p=dropout))
        if dropout > 0:
            self.mlp = Sequential(*layers[:-2])
        else:
            self.mlp = Sequential(*layers[:-1])

    def forward(self, x):
        return self.mlp(x)


class ForwardForwardNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dims, device, dropout=0.0):
        super().__init__()

        hidden_dims = [input_dim] + hidden_dims
        self.layers = []
        for (in_dim, out_dim) in zip(hidden_dims[:-1], hidden_dims[1:]):
            self.layers += [Layer(in_dim, out_dim, device=device, dropout=dropout)]

    def predict(self, x):
        h = x
        goodness = []
        for layer in self.layers:
            h = layer(h)
            goodness += [h.pow(2).mean(1)]

        return sum(goodness).unsqueeze(1)

    def ff_train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            h_pos, h_neg = layer.train(h_pos, h_neg)


class Layer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, device, dropout=0, threshold=2.0, opt_iters=10, lr=0.001, eps=1e-4, classif=True):
        super().__init__()
        self.classif = classif
        self.threshold = threshold
        self.opt_iters = opt_iters

        layers = []
        layers.append(Linear(in_dim, out_dim))
        layers.append(ReLU())
        if dropout > 0:
            layers.append(Dropout(p=dropout))

        if dropout > 0:
            self.mlp = Sequential(*layers[:-2])
        else:
            self.mlp = Sequential(*layers[:-1])
        self.to(device)
        self.optimizer = Adam(self.parameters(), lr=lr)
        self.eps = eps

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 1e-4)
        return self.mlp(x_direction)

    def train(self, x_pos, x_neg):
        for i in range(self.opt_iters):
            g_pos = self.forward(x_pos).pow(2).mean(1)
            g_neg = self.forward(x_neg).pow(2).mean(1)
            # BCE w/o constant factors (calculated from definition)
            loss = torch.log(1 + torch.exp(torch.cat([-g_pos + self.threshold, g_neg - self.threshold]))).mean()

            self.optimizer.zero_grad()
            # this backward just compute the derivative and hence is not considered backpropagation
            loss.backward()
            self.optimizer.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()