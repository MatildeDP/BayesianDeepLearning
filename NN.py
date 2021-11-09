import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import plot_decision_boundary
from copy import copy



# TODO implement running average instead of storing lists for loss!!!


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.tanh = nn.Tanh()
        #self.s = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        #ss = self.s(x)
        return x


    def predict(self, X):
        outs = self(X)
        s = nn.Softmax(dim=1)
        probs = s(outs)

        return probs, torch.max(probs, 1).indices, outs



