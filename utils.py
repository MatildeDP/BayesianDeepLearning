import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from sklearn.datasets import make_moons
from matplotlib.colors import ListedColormap

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

def plot_weight_and_loss(model, loss,  epoch  = 'No epochs'):
    params = list(model.parameters())
    fig, axes = plt.subplots(2,3, figsize=(10, 5))
    fig.suptitle('Weights and biases og epoch %s' %epoch)
    axes[0,0].plot(params[0].detach().numpy(), '.')
    axes[0,1].plot(params[1].detach().numpy(), '.')
    axes[1,0].plot(params[2].detach().numpy(), '.')
    axes[1,1].plot(params[3].detach().numpy(), '.')
    sns.lineplot(ax = axes[1,2], data = pd.DataFrame(loss), palette="tab10")

    plt.show()


def decision_boundary(X, y, yhat):
    #TODO: make this one
    return


if __name__== '__main__':
    data = make_moons(n_samples=100, shuffle=True, noise=False, random_state=None)
    X, y = data[0], data[1]
    X = torch.tensor(X, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)

    plt.show()
