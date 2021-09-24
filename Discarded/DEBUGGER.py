from sklearn.datasets import make_moons
import torch
from Network import DeterministicNet
from copy import deepcopy
import matplotlib.pyplot as plt
from Data import Data, DataLoaderInput
from torch.utils.data import DataLoader
from Model import Test, Train
from utils import decision_boundary, load_net
import torch.nn as nn
import torch.optim as optim




if __name__ == '__main__':
    num_epochs = 40
    # Load data
    D = Data('twomoons', n_test_samples=100, n_train_samples=1000, noise=0.05, load=True)
    D.load_train_values()
    D.load_test_values()

    # Prepare class instance for torch dataloader
    TrainSet = DataLoaderInput(D.Xtrain, D.ytrain)
    TestSet = DataLoaderInput(D.Xtest, D.ytest)

    # Define iter
    TrainLoader = DataLoader(TrainSet, batch_size=128, shuffle=True, num_workers=2)
    TestLoader = DataLoader(TestSet, batch_size=32, shuffle=True, num_workers=2)

    # Define/Load Network


    net = DeterministicNet()
    criterion_ = nn.BCELoss()

    #net = DeterministicNet(in_dim=2, out_dim=1, n_dim=30)
    #criterion_ = nn.NLLLoss()

    lr = 0.01
    optimizer_ = optim.SGD(net.parameters(), lr=lr)

    # Train Network
    Train(model = net, num_epochs = num_epochs, trainloader = TrainLoader, optimizer = optimizer_, criterion = criterion_, save_path='models/NN_BCE1.pth', save=True, plot_each_epoch=False)
    # Test Network
    net_trained = load_net(DeterministicNet(), 'models/NN_BCE1.pth')
    acc, yhat, preds = Test(net_trained, TestLoader, return_all=True, plot_decision_boundary=False)

    decision_boundary(model = net_trained, dataloader = TestLoader, y_hat=0)
    #a = 123



    #decision_boundary(X, y, y_hat, model)
