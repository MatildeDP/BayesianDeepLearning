
# https://www.datahubbs.com/deep-learning-101-first-neural-network-with-pytorch/
#overnstående tester forskellige kombinationer af hiddens nodes
import torch.optim as optim
from Data import Data
from torch.utils.data import DataLoader
import torch.nn as nn


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from copy import deepcopy, copy

def scatter(X, y, yhat, epoch):
    sns.set_theme()

    # Data
    X_ = X.detach().numpy()
    y_ = y.detach().numpy()
    yhat_ = [1 if x > 0.5 else 0 for x in yhat.detach().numpy()]

    # set axis
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    fig.suptitle('Scatterplot, epoch %s' %epoch)
    axes[0].set_title('Labels')
    axes[1].set_title('Yhat')

    sns.scatterplot(ax = axes[0], x=X_[:, 0], y=X_[:, 1], hue=y_, style = y_)
    sns.scatterplot(ax=axes[1], x=X_[:, 0], y=X_[:, 1], hue = yhat_, style = yhat_)
    axes[2].plot(yhat.detach().numpy(), '.')
    plt.show()


def Train(model, num_epochs, trainloader, optimizer, criterion, save_path='', save=False, plot_each_epoch=False):
    all_epoch_average = {}
    all_epoch_loss = {}

    # training mode
    model.train()

    for epoch in range(num_epochs):  # iterate through dataset num_epoch times
        print(epoch)
        running_loss = 0  # keep track of loss
        epoch_loss = {'Running Average': [], 'Loss': []}


        for batch_idx, (X,y,idx) in enumerate(trainloader):  # iterate through trainloader

            y = y.float()

            # Zero gradients
            optimizer.zero_grad()

            # Forward
            out_ = model(X)
            out_ = torch.reshape(out_,(-1,))
            out_.float()

            # Loss
            loss = criterion(out_, y)
            epoch_loss['Loss'].append(loss.item())

            # Backward
            loss.backward()

            # Optimize
            optimizer.step()

            # print
            running_loss += loss.item()
            epoch_loss['Running Average'].append(running_loss / (batch_idx + 1))
            # print('Running average loss: %s' % (running_loss / (idx + 1)))

        # plot scatter
        if epoch%20 == 0:
            scatter(X, y, out_, epoch)


        # plot loss and weights
        #if plot_each_epoch:
            #plot_weight_and_loss(model, epoch_loss, epoch)

        # save  model
        if save and epoch == (num_epochs - 1):
            torch.save(model.state_dict(), save_path)

        all_epoch_loss[epoch] = epoch_loss['Loss']
        all_epoch_average[epoch] = epoch_loss['Running Average']

    sns.lineplot(data=pd.DataFrame(all_epoch_average), palette="tab10")
    plt.show()
    sns.lineplot(data=pd.DataFrame(all_epoch_loss), palette="tab10")
    plt.show()

    return


def Test(model, testloader, return_all=False, plot_decision_boundary=False):
    '''
    :param newmodel: input new model if you wish to test a saved model
    :return:
    '''
    acc = 0
    yhat_= torch.tensor([])
    model.eval()
    with torch.no_grad():

        for batch_idx, batch in enumerate(testloader):
            X, y = batch[:2]
            yhat, preds = model.predict(X)

            if return_all and batch_idx == 0:
                yhat_ = torch.cat((yhat_, yhat))
                preds_ = copy(preds)

            elif return_all:
                yhat_ = torch.cat((yhat_, yhat))
                preds_ = torch.cat((preds_,  preds)) # np.append(preds_,preds, axis = 0)

            #new_acc = 1 - sum(abs(yhat - y)) / len(yhat)

            #acc = acc*(idx)/(idx + 1) + new_acc/(idx+1)
            acc =0

            # scatter(X, y, yhat, idx)

    model.train()
    if return_all:
        return acc, yhat_, preds_
    else:
        return acc
