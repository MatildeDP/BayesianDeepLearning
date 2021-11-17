import torch
import torch.nn as nn
from deterministic import Deterministic_net
from data import DataLoaderInput
from utils import plot_decision_boundary, plot_acc_and_loss, compare_parameter_loss_plot
from sklearn.datasets import make_moons, make_circles
from SWAG import Swag
import seaborn as sns
import matplotlib.pyplot as plt


def main_deterministic(batch_size, num_epochs, hidden_dim, learning_rate, noise, net_path, opti_path, momentum, gamma,
                       load_pretrained=True, save_model=False, input_dim=2, output_dim=2):
    # Print information
    print(
        "Running deterministic network with following settings: \n Batch size: %s  \n Number of epochs: %s \n Network dimensions: %s" % (
            batch_size, num_epochs, (input_dim, hidden_dim, output_dim)))
    print("Learning rate: %s" % learning_rate)
    print("Using pretrained network: %s" % load_pretrained)
    print("Saving network: %s" % save_model)
    print("Save paths: %s and %s" % (net_path, opti_path))

    # Data
    train_dataset = make_moons(n_samples=1000, noise=noise, random_state=3)
    Xtrain, ytrain = train_dataset

    test_dataset = make_moons(n_samples=200, noise=noise, random_state=3)
    Xtest, ytest = test_dataset

    train_loader = torch.utils.data.DataLoader(dataset=DataLoaderInput(Xtrain, ytrain),
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=DataLoaderInput(Xtest, ytest),
                                              batch_size=batch_size,
                                              shuffle=False)

    # Define model etc.
    model = Deterministic_net(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=5 * 10 ** (-4),
                                nesterov=False)

    # Load pretrained
    if load_pretrained:
        model.load_state_dict(torch.load(net_path))
        optimizer.load_state_dict(torch.load(opti_path))

    # Train and test
    train_loss, test_loss, acc = model.run(num_epochs, test_loader, train_loader, criterion, optimizer,
                                           save_net=save_model, net_path=net_path,
                                           opti_path=opti_path)

    # Final plots
    decision_path = "/Deterministic/Decision_hidden" + str(hidden_dim) + "_Lr" + str(learning_rate) + "_noise" + str(
        noise) + ".jpg"
    loss_path = "/Deterministic/Loss_accu_hidden" + str(hidden_dim) + "_Lr" + str(learning_rate) + "_noise" + str(
        noise) + ".jpg"
    plot_decision_boundary(model, dataloader = test_loader, S = 20, title="Decision boundary with test points on trained model",
                           save_image_path=decision_path)
    plot_acc_and_loss(testloss=test_loss, trainloss=train_loss, accuracy=acc, save_path=loss_path)


def main_swag(batch_size, num_epochs, hidden_dim, learning_rate, c, K, S, noise, load_net_path, load_opti_path,
              l2_param, C,
              input_dim=2, output_dim=2, plot=True):
    print(
        "Running swag network with following settings: \n Batch size: %s  \n Number of epochs: %s \n Network dimensions: %s" % (
            batch_size, num_epochs, (input_dim, hidden_dim, output_dim)))
    print(" Learning rate: %s" % learning_rate)
    print(" Moment update frequency: %s" % c)
    print(" Number of columns in deviation matrix Dhat: %s" % K)
    print(" Save paths: %s and %s" % (load_net_path, load_opti_path))
    print(" Number of models to sample from: %s" % S)

    # Load data
    train_dataset = make_moons(n_samples=1000, noise=noise, random_state=3)
    Xtrain, ytrain = train_dataset
    Xtrain = torch.tensor(Xtrain, dtype=torch.float)
    ytrain = torch.tensor(ytrain, dtype=torch.long)

    test_dataset = make_moons(n_samples=200, noise=noise, random_state=3)
    Xtest, ytest = test_dataset
    Xtest = torch.tensor(Xtest, dtype=torch.float)
    ytest = torch.tensor(ytest, dtype=torch.long)

    train_loader = torch.utils.data.DataLoader(dataset=DataLoaderInput(Xtrain, ytrain),
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=DataLoaderInput(Xtest, ytest),
                                              batch_size=batch_size,
                                              shuffle=False)

    # Define model etc.
    criterion = nn.CrossEntropyLoss()
    model = Swag(input_dim, hidden_dim, output_dim, K, c=c, S=S, criterion=criterion, num_epochs=num_epochs,
                 learning_rate=learning_rate, l2_param=l2_param, C=C)

    # Train and test
    train_loss, test_loss, accuracy = model.run_swag(net_path=load_net_path, opti_path=load_opti_path,
                                                     train_loader=train_loader,test_loader = test_loader, Xtest=Xtest, ytest=ytest)

    if plot:
        # Final plots
        decision_path = "SWAG/Decision_K" + str(K) + '_c' + str(c) + "_batch" + str(batch_size) + "_epochs" + str(
            num_epochs) + "_noise" + str(noise) + "_lr" + str(learning_rate) + ".jpg"
        plot_decision_boundary(model, test_loader, S=S,
                               title="Final SWAG with test points", predict_func='stochastic',
                               save_image_path=decision_path)

        loss_path = "SWAG/Loss_accu_K" + str(K) + '_c' + str(c) + "_batch" + str(batch_size) + "_epochs" + str(
            num_epochs) + "_noise" + str(noise) + "_lr" + str(learning_rate) + ".jpg"
        plot_acc_and_loss(testloss=test_loss, trainloss=train_loss, accuracy=accuracy, save_path=loss_path)

        return train_loss, test_loss, model

    else:
        return train_loss, test_loss, model





if __name__ == '__main__':

    run = 'swag'

    # Parameters
    """
    batch_size = 16 # [8, 16, 32, 64, 128] 8 eller 16!!
    num_epochs = 1000
    input_dim = 2
    hidden_dim = 50
    output_dim = 2
    learning_rate = 1.3
    noise = 0.3    """

    batch_size = 8
    num_epochs = 100
    input_dim = 2
    hidden_dim = 50
    output_dim = 2
    learning_rate = 1
    noise = 0.2
    momentum = 0
    gamma = 0.02  # decay rate
    l2_param = 0.001
    C = 2

    net_path = 'models/NN_50.pth'
    opti_path = 'Optimizers/Opti_50.pth'

    # Parameters for deterministic
    load_pretrained = False
    save_model = True

    # Parameters for SWAG
    c = 1  # dhat update freq
    K = 20  # dim of dhat
    S = 20  # number of settings bma

    # Run deterministic net
    if run == 'deterministic':
        main_deterministic(batch_size, num_epochs, hidden_dim, learning_rate, noise, net_path, opti_path, gamma=gamma,
                           momentum=momentum,
                           load_pretrained=load_pretrained, save_model=save_model)

    # Run SWAG
    elif run == 'swag':
        main_swag(batch_size, num_epochs, hidden_dim, learning_rate, c, K,
                  S, noise, net_path, opti_path, l2_param=l2_param, C = C)
