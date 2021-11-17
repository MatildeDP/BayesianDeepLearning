import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from copy import deepcopy
from BMA import monte_carlo_bma
from scipy.linalg import fractional_matrix_power


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


def plot_acc_and_loss(testloss, trainloss, accuracy, save_path, save_image_path = True):
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

    if save_image_path:
        plt.savefig("Plots/" + save_path)

    plt.show()
    plt.close()

def compare_parameter_loss_plot(trainloss, testloss, param, num_epochs, save_path = ''):
    """
    :param trainloss: dict: key = parameter value, value: loss
    :param testloss: dict: key = parameter value, value: loss
    :param: param: str: which parameter
    :return:
    """
    sns.set_theme()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Loss for different settings of' + param)

    trainloss['Epochs'] = [i for i in range(num_epochs)]
    testloss['Epochs'] = [i for i in range(num_epochs)]

    df_trainloss = pd.DataFrame(trainloss)
    df_testloss = pd.DataFrame(testloss)


    sns.lineplot(ax=axes[0], x='Epochs', y='value', hue='variable', data=pd.melt(df_trainloss, 'Epochs'),palette = "tab10" )
    sns.lineplot(ax=axes[1], x='Epochs', y='value', hue='variable', data=pd.melt(df_testloss, 'Epochs'), palette = "tab10")


    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')

    axes[1].set_title('Test Loss')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')

    if save_path:
        plt.savefig("Plots/" + save_path)

    plt.show()



def plot_decision_boundary(model, dataloader, S, title="", predict_func = 'predict', save_image_path = ""):
    X = dataloader.dataset.X.clone().detach()
    y = dataloader.dataset.y.clone().detach()

    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - 2, X[:, 0].max() + 2
    y_min, y_max = X[:, 1].min() - 2, X[:, 1].max() + 2
    h = 0.01

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole gid
    if predict_func == 'predict':
        probs_grid, _, _ = model.predict(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32))

    elif predict_func == 'stochastic':
        #probs_grid, _, _ = bma(model, S = 20, Xtest = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32), ytest = 0, criterion = 0, test = False)
        probs_grid = monte_carlo_bma(model, Xtest=torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float64), ytest=0, S=S, C=2, forplot=True)
    #elif predict_func == 'kfac':
        #probs_grid = model.monte_carlo_bma(Xtest = torch.tensor(np.c_[xx.ravel(), yy.ravel()]), ytest=0, S = 20, C = 2 ,forplot = True)

    #probs_grid = monte_carlo_bma(model, Xtest = X, ytest = y, S = S, C = 2, forplot=True)

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


    if save_image_path:
        plt.savefig("Plots/" + save_image_path)

    plt.show()
    plt.close()





def Squared_matrix(D, is_diagonal = True):
    """
    params: D: if is_diagonal is false is needs a 2D vector, if is_diagonal is true it needs a 1D vector
    """

    if is_diagonal:
        # test if D is positive semi definite
        if np.any(D.detach().numpy() < 0):
            print("Some eigenvalues are smaller than 0")
            return False
        else:
            # principal root of D
            new = torch.sqrt(D)
            return new



