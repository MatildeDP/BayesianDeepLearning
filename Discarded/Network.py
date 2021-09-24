import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import plot_weight_and_loss
import numpy as np


# code from https://medium.com/@prudhvirajnitjsr/simple-classifier-using-pytorch-37fba175c25c
# loss = nn.BCELoss()
# our class must extend nn.Module

class DeterministicNet(nn.Module):
    """
    Fully connected NN with two linear layers
    """

    def __init__(self):
        super(DeterministicNet, self).__init__()
        self.fc1 = nn.Linear(2,30)  # ax+b
        self.fc2 = nn.Linear(30,40)  # ax+b
        self.fc3 = nn.Linear(40,1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)  # activation function
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        #x = F.log_softmax(x, dim = 1) # activation function
        return x

    # This function takes an input and predicts the class, (0 or 1)
    def predict(self, X):
        # Apply softmax to output.
        out = self.forward(X)
        #pred = np.exp(out.detach().numpy())
        ans = []
        # Pick the class with maximum weight
        for t in out:
            #if t[0] > t[1]:
            #    ans.append(0)
            #else:
            #    ans.append(1)
            if t> 0.5:
                ans.append(1)
            else:
                ans.append(0)
        return torch.tensor(ans), out



