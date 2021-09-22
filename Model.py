from NewNetwork import DeterministicNet
from Data import DataLoaderInput
import torch.optim as optim
from Data import Data
from torch.utils.data import DataLoader
import torch.nn as nn
from plots import plot_weight_and_loss, scatter
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class Network:
    def __init__(self, model, trainloader, testloader, criterion, optimizer, lr=0.001):

        self.trainloader = trainloader
        self.testloader = testloader

        if model.lower() == 'deterministicnet':
            self.model = DeterministicNet()

        if criterion.lower() == 'nll':
            self.criterion = nn.NLLLoss()

        if optimizer.lower() == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr)

    def train_net(self, num_epochs, save=False, plot_each_epoch=False):
        all_epoch_average = {}
        all_epoch_loss = {}

        for epoch in range(num_epochs):  # iterate through dataset num_epoch times
            print(epoch)
            running_loss = 0  # keep track of loss
            epoch_loss = {'Running Average': [], 'Loss': []}

            for idx, batch in enumerate(self.trainloader, 0):  # iterate through trainloader

                X, y = batch[:2]

                # Zero gradients
                self.optimizer.zero_grad()

                # Forward
                out_ = self.model(X)

                # Loss
                loss = self.criterion(out_, y)
                epoch_loss['Loss'].append(loss.item())

                # Backward
                loss.backward()

                # Optimize
                self.optimizer.step()

                # print
                running_loss += loss.item()
                epoch_loss['Running Average'].append(running_loss / (idx + 1))
                #print('Running average loss: %s' % (running_loss / (idx + 1)))

            # plot moons


            # plot loss and weights
            if plot_each_epoch:
                plot_weight_and_loss(self.model, epoch_loss, epoch)

            # TODO: Save model

            all_epoch_loss[epoch] = epoch_loss['Loss']
            all_epoch_average[epoch] = epoch_loss['Running Average']

        sns.lineplot(data=pd.DataFrame(all_epoch_average), palette="tab10")
        plt.show()
        sns.lineplot(data=pd.DataFrame(all_epoch_loss), palette="tab10")
        plt.show()


        return

    def Test(self, newmodel=''):

        '''
        :param newmodel: input new model if you wish to test a saved model
        :return:
        '''
        if newmodel:
            self.model = newmodel
        else:
            pass
        acc_ = 0

        for idx, batch in enumerate(self.testloader):
            X, y = batch[:2]
            yhat = self.model.predict(X)
            acc_ += sum(abs(yhat - y)) / len(yhat)

            scatter(X, y, yhat, idx)
        acc = 1 - acc_/(idx + 1)
        return acc


if __name__ == '__main__':

    num_epochs = 20
    # Load data
    D = Data('twomoons', n_test_samples=100,n_train_samples=1000, noise=0.001, load=True)
    D.load_train_values()
    D.load_test_values()

    # Prepare class instance for torch dataloader
    TrainSet = DataLoaderInput(D.Xtrain, D.ytrain)
    TestSet = DataLoaderInput(D.Xtest, D.ytest)

    # Define iter
    TrainLoader = DataLoader(TrainSet, batch_size=64, shuffle=True, num_workers=2)
    TestLoader = DataLoader(TestSet, batch_size=32, shuffle=True, num_workers=2)

    # Define Network
    model = Network(model='deterministicnet', trainloader=TrainLoader,testloader = TestLoader, criterion='nll', optimizer='SGD')

    # Train Network
    model.train_net(num_epochs=num_epochs)

    #Test Network
    a = model.Test()
    print(a)
    b = 123



