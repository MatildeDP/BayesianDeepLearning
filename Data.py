from matplotlib import pyplot as plt
from sklearn.datasets import make_moons
import numpy as np
from torch.utils.data import DataLoader
import torch


class Data:
    def __init__(self, name, n_test_samples,n_train_samples, noise=False, shuffle=True, load = True):
        self.name = name
        self.n_test_samples = n_test_samples
        self.n_train_samples = n_train_samples
        self.noise = noise
        self.shuffle = shuffle
        self.Xtest = 0
        self.ytest = 0
        self.Xtrain = 0
        self.ytrain = 0
        self.load = load


    def load_test_values(self):
        if self.name.lower() == 'twomoons':
            test = make_moons(n_samples=self.n_test_samples, shuffle=self.shuffle, noise=self.noise, random_state=None)
        self.Xtest = test[0]
        self.ytest = test[1]

    def load_train_values(self):
        if self.name.lower() == 'twomoons':
            train = make_moons(n_samples=self.n_train_samples, shuffle=self.shuffle, noise=self.noise, random_state=None)
        self.Xtrain = train[0]
        self.ytrain = train[1]


class DataLoaderInput:
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.tensor(self.y[idx])
        return x, y, idx

    def __len__(self):
        return len(self.X)
