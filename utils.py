import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from Model import Test, Train
from Data import DataLoaderInput
from torch.utils.data import DataLoader


def save_net(path, model):
    torch.save(model.state_dict(), path)


def load_net(class_instance, path):
    """
    :param class_instance: Instance of NN class
           path: path were NN is saved
    :return:
    """
    model_ = class_instance
    model_.load_state_dict(state_dict=torch.load(path))
    model_.eval()  # set model to eval mode
    return deepcopy(model_)


def scatter(X, y, yhat, epoch):
    sns.set_theme()

    # Data
    X_ = X.detach().numpy()
    y_ = y.detach().numpy()
    yhat_ = [1 if x > 0.5 else 0 for x in yhat.detach().numpy()]

    # set axis
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    fig.suptitle('Scatterplot, epoch %s' % epoch)
    axes[0].set_title('Labels')
    axes[1].set_title('Yhat')

    sns.scatterplot(ax=axes[0], x=X_[:, 0], y=X_[:, 1], hue=y_, style=y_)
    sns.scatterplot(ax=axes[1], x=X_[:, 0], y=X_[:, 1], hue=yhat_, style=yhat_)
    axes[2].plot(yhat.detach().numpy(), '.')
    plt.show()


def plot_acc_and_loss(testloss, trainloss, accuracy):
    sns.set_theme()
    num_epochs = len(accuracy)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Loss and Accuracy')

    dict_loss = {'Test': testloss,  'Train': trainloss, 'Epochs': [i for i in range(num_epochs)]}
    df_loss = pd.DataFrame(dict_loss)

    dict_acc = {'Accuracy':  accuracy, 'Epochs': [i for i in range(num_epochs)]}
    df_acc = pd.DataFrame(dict_acc)


    sns.lineplot(ax = axes[0], x = 'Epochs', y = 'value', hue='variable',  data=pd.melt(df_loss, 'Epochs'))
    axes[1].plot([i for i in range(num_epochs)],accuracy)
    axes[0].set_xlabel('Epochs')
    axes[1].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[1].set_ylabel('Accuracy')

    plt.show()


def plot_decision_boundary(model, X, y, title=""):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole gid
    probs_grid, _, _ = model.predict(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32))
    probs0_grid = probs_grid.detach().numpy()[:, 1]
    probs0_grid = probs0_grid.reshape(xx.shape)

    # Plot the contour and test examples
    contour = plt.contourf(xx, yy, probs0_grid, cmap='RdBu')
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='RdBu')
    ax = plt.colorbar(contour)
    ax.set_label("$P(y = 1)$")
    ax.set_ticks([0, .25, .5, .75, 1])
    # plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu')
    plt.xlabel = ("$X_1$")
    plt.ylabel = ("$X_2$")
    plt.title(title)

    plt.show()
