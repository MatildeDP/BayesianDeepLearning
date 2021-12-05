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
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)


    def forward(self, x):

        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x

    def predict(self, X):
        s = nn.Softmax(dim=1)
        score = self(X)
        probs = s(score)
        pred = torch.max(probs, 1).indices

        return probs, score, pred


