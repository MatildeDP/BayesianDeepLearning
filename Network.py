import torch
import torch.nn as nn
import torch.nn.functional as F
from plots import plot_weight_and_loss
import numpy as np


# code from https://medium.com/@prudhvirajnitjsr/simple-classifier-using-pytorch-37fba175c25c
# loss = nn.BCELoss()
# our class must extend nn.Module

class DeterministicNet(nn.Module):

    def __init__(self):
        super(DeterministicNet, self).__init__()
        self.fc1 = nn.Linear(2, 3)  # ax+b
        self.fc2 = nn.Linear(3, 2)  # ax+b

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)  # activation function
        x = self.fc2(x)
        x = F.log_softmax(x, dim = 1)
        return x

    # This function takes an input and predicts the class, (0 or 1)
    def predict(self, X):
        # Apply softmax to output.
        out = self.forward(X)
        pred = np.exp(out.detach().numpy())
        ans = []
        # Pick the class with maximum weight
        for t in pred:
            if t[0] > t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)



