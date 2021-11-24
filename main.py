import torch
import torch.nn as nn
from deterministic import Deterministic_net
from data import DataLoaderInput, LoadDataSet
from utils import plot_decision_boundary, plot_acc_and_loss, compare_parameter_loss_plot

from SWAG import Swag
import seaborn as sns
import matplotlib.pyplot as plt


def main_deterministic(batch_size, num_epochs, hidden_dim, input_dim, output_dim, learning_rate, net_path, opti_path,
                       data, loaders, load_pretrained=True, save_model=False):
    # Print information
    print(
        "Running deterministic network with following settings: \n Batch size: %s  \n Number of epochs: %s \n Network dimensions: %s" % (
            batch_size, num_epochs, (input_dim, hidden_dim, output_dim)))
    print("Learning rate: %s" % learning_rate)
    print("Using pretrained network: %s" % load_pretrained)
    print("Saving network: %s" % save_model)
    print("Save paths: %s and %s" % (net_path, opti_path))


    Xtrain, ytrain, Xtest, ytest = data
    train_loader, test_loader = loaders

    # Define model etc.
    model = Deterministic_net(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=5 * 10 ** (-4),
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
    #decision_path = "/Deterministic/Decision_hidden" + str(hidden_dim) + "_Lr" + str(learning_rate)+ ".jpg"
    loss_path = "/Deterministic/Loss_accu_MNIST_hidden" + str(hidden_dim) + "_Lr" + str(learning_rate) +".jpg"
    #plot_decision_boundary(model, dataloader = test_loader, S = 20, title="Decision boundary with test points on trained model",
                           #save_image_path=decision_path)
    plot_acc_and_loss(testloss=test_loss, trainloss=train_loss, accuracy=acc, save_path=loss_path)


def main_swag(batch_size, num_epochs, hidden_dim, learning_rate, c, K, S,data, loaders, net_path, opti_path,
              l2_param, input_dim, output_dim, plot=True):
    print(
        "Running swag network with following settings: \n Batch size: %s  \n Number of epochs: %s \n Network dimensions: %s" % (
            batch_size, num_epochs, (input_dim, hidden_dim, output_dim)))
    print(" Learning rate: %s" % learning_rate)
    print(" Moment update frequency: %s" % c)
    print(" Number of columns in deviation matrix Dhat: %s" % K)
    print(" Save paths: %s and %s" % (net_path, opti_path))
    print(" Number of models to sample from: %s" % S)

    Xtrain, ytrain, Xtest, ytest = data
    train_loader, test_loader = loaders

    # Define model etc.
    criterion = nn.CrossEntropyLoss()
    model = Swag(input_dim, hidden_dim, output_dim, K, c=c, S=S, criterion=criterion, num_epochs=num_epochs,
                 learning_rate=learning_rate, l2_param=l2_param)

    # Train and test
    train_loss, test_loss, accuracy = model.run_swag(net_path=net_path, opti_path=opti_path,
                                                     train_loader=train_loader,test_loader = test_loader, Xtest=Xtest, ytest=ytest)

    if plot:
         #Final plots
        decision_path = "SWAG/Decision_K" + str(K) + '_c' + str(c) + "_batch" + str(batch_size) + "_epochs" + str(
            num_epochs) + "_lr" + str(learning_rate) + ".jpg"
        plot_decision_boundary(model, test_loader, S=S,
                               title="Final SWAG with test points", predict_func='stochastic',
                               save_image_path=decision_path)

        loss_path = "SWAG/Loss_accu_MNIST_K" + str(K) + '_c' + str(c) + "_batch" + str(batch_size) + "_epochs" + str(
            num_epochs) + "_lr" + str(learning_rate) + ".jpg"
        plot_acc_and_loss(testloss=test_loss, trainloss=train_loss, accuracy=accuracy, save_path=loss_path)

        return train_loss, test_loss, model

    else:
        return train_loss, test_loss, model


def settings(data):
    if data == 'two_moons':
        input_dim = 2
        output_dim = 2

    if data == 'mnist':
        input_dim = 784
        output_dim = 10

    return input_dim, output_dim



if __name__ == '__main__':

    # Get mnist back in shape reshape(-1,1,28,28), reshape(-1, 28*28)

    run = 'swag'
    which_data = 'mnist'

    # Parameters
    """
    batch_size = 16 # [8, 16, 32, 64, 128] 8 eller 16!!
    num_epochs = 1000
    input_dim = 2
    hidden_dim = 50
    output_dim = 2
    learning_rate = 1.3
    noise = 0.3    """

    batch_size = 128
    num_epochs = 11 # må ikke være mindre end K for SWAG
    hidden_dim = 100
    learning_rate = 0.1
    l2_param = 0.001

    #net_path = 'models/two_moons/NN_5.pth'
    #opti_path = 'Optimizers/two_moons/Opti_5.pth'

    net_path = 'models/MNIST/NN_100.pth'
    opti_path = 'Optimizers/MNIST/Opti_100.pth'

    # Parameters for deterministic
    load_pretrained = False
    save_model = True

    # Parameters for SWAG
    c = 1  # dhat update freq
    K = 10  # dim of dhat
    S = 5  # number of settings bma

    # Load data
    Dataset = LoadDataSet(which_data)
    Xtrain, ytrain, train_loader = Dataset.load_training_data(batch_size = batch_size, noise = 0.3)
    Xtest, ytest, test_loader = Dataset.load_test_data(batch_size = batch_size, noise=0.3)

    data = (Xtrain, ytrain, Xtest, ytest)
    loaders = (train_loader, test_loader)

    # Define in- and output dims
    input_dim, output_dim = settings(which_data)

    # Run deterministic net
    if run == 'deterministic':
        main_deterministic(batch_size = batch_size, num_epochs=num_epochs, hidden_dim=hidden_dim,
                           input_dim = input_dim, output_dim = output_dim, learning_rate = learning_rate,
                           net_path = net_path, opti_path = opti_path,
                           data = data, loaders = loaders, load_pretrained=load_pretrained, save_model=save_model)

    # Run SWAG
    elif run == 'swag':
        main_swag(batch_size = batch_size, num_epochs = num_epochs, hidden_dim = hidden_dim, learning_rate = learning_rate,
                  c = c, K=K, S = S, data = data, loaders= loaders,net_path = net_path, opti_path = opti_path,
                  l2_param=l2_param, input_dim = input_dim, output_dim=output_dim)

