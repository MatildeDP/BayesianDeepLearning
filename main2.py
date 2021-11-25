import torch
import torch.nn as nn
from deterministic import Deterministic_net
from data import DataLoaderInput, LoadDataSet
from utils import plot_decision_boundary, plot_acc_and_loss, compare_parameter_loss_plot

from SWAG import Swag
from BMA import monte_carlo_bma


def train_deterministic(which_data: str, pretrained: bool, save_net = True):
    # Define param space
    param_space = {'device': "cuda" if torch.cuda.is_available() else "cpu",
                   'train_batch_size': 64,
                   'test_batch_size': 1000,
                   'n_epochs': 1,
                   'seed': 0,
                   'hidden_dim': 1024,
                   'save_model': False,
                   'lr': 0.001,
                   'l2': 0.001,
                   'momentum': 0.9}

    # set seed
    torch.manual_seed(param_space['seed'])

    # get dims
    input_dim, output_dim = settings(which_data)

    # Load data
    Dataset = LoadDataSet(which_data)
    Xtrain, ytrain, train_loader = Dataset.load_training_data(batch_size=batch_size, noise=0.3)
    Xtest, ytest, test_loader = Dataset.load_test_data(batch_size=batch_size, noise=0.3)

    # Define model, optimizer, criterion
    model = Deterministic_net(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=param_space['lr'],
                                weight_decay=param_space['l2'],
                                momentum=param_space['momentum'],
                                nesterov=False)

    # Load pretrained
    if pretrained:
        model.load_state_dict(torch.load(net_path))
        optimizer.load_state_dict(torch.load(opti_path))
        for g in optimizer.param_groups:
            g['weight_decay'] = l2_param
            g['lr'] = learning_rate
            g['momentum'] = momentum

    # Train model
    test_loss, train_loss, acc = [], [], []
    for epoch in range(param_space['n_epochs']):

        # One epoch of training
        optimizer, criterion, train_loss_ = model.train_net(train_loader, optimizer, criterion,
                                                            save_net=save_net, net_path=net_path, opti_path=opti_path)
        # Test
        accuracy, tst_loss, _ = model.test_net(test_loader=test_loader, criterion=criterion,
                                               freq=30,  epoch=epoch)
        # Collect loss
        train_loss.append(float(train_loss_))
        test_loss.append(float(tst_loss.detach().numpy()))
        acc.append(accuracy)

        if epoch % 1 == 0:
            print('Iteration: {}. Test Loss: {}. Train Loss: {}. Accuracy: {}'.format(epoch, tst_loss.item(),
                                                                                      train_loss_, accuracy))
    # Final plots
    if which_data == 'two_moons':
        decision_path = "/Deterministic/Decision_hidden" + str(hidden_dim) + "_Lr" + str(learning_rate) + ".jpg"
        plot_decision_boundary(model, dataloader=test_loader, S=20,
                               title="Decision boundary with test points on trained model",
                               save_image_path=decision_path)

    loss_path = "/Deterministic/Loss_accu_MNIST_hidden" + str(hidden_dim) + "_Lr" + str(learning_rate) + ".jpg"
    plot_acc_and_loss(testloss=test_loss, trainloss=train_loss, accuracy=acc, save_path=loss_path)



def train_swag(which_data: str, pretrained: bool, save_net = True):

    param_space = {'device': "cuda" if torch.cuda.is_available() else "cpu",
                   'train_batch_size': 64,
                   'test_batch_size': 1000,
                   'n_epochs': 1,
                   'seed': 0,
                   'hidden_dim': 1024,
                   'save_model': False,
                   'lr': 0.001,
                   'l2': 0.001,
                   'momentum': 0.9, 'c': 1,
                   'K': 10,
                   'S': 5}

    # set seed
    torch.manual_seed(param_space['seed'])

    # Define in- and output dims
    input_dim, output_dim = settings(which_data)

    # Load data
    Dataset = LoadDataSet(which_data)
    Xtrain, ytrain, train_loader = Dataset.load_training_data(batch_size=batch_size, noise=0.3)
    Xtest, ytest, test_loader = Dataset.load_test_data(batch_size=batch_size, noise=0.3)

    # Define model, optimizer, criterion
    criterion = nn.CrossEntropyLoss()
    model = Swag(input_dim = input_dim, hidden_dim = param_space['hiden_dim'], output_dim = output_dim,
                 K = param_space['K'], c=param_space['c'], S=param_space['S'],
                 criterion=criterion, num_epochs=param_space['n_epochs'],
                 learning_rate=param_space['lr'], l2_param=param_space['l2'])

    optimizer = torch.optim.SGD(model.parameters())

    # Load pretrained
    model.load_state_dict(torch.load(net_path))
    optimizer.load_state_dict(torch.load(opti_path))

    for g in optimizer.param_groups:
        g['lr'] = param_space['lr'],
        g['weight_decay'] = param_space['l2'],
        g['momentum'] = param_space['momentum'],
        g['nesterov'] = False

    # Train and test
    # Extract initial weights
    theta1, theta2 = model.get_ith_moments()

    # Initialize error and accuracy container
    test_loss, train_loss, all_acc = [], [], []
    n = 0

    # Run T epoch of training and testing
    for i in range(param_space['n_epochs']):

        # Train swag
        n, theta1, theta2, loss_ = model.train_swag(epoch=i, n=n, theta1=theta1, theta2=theta2,
                                                   train_loader=train_loader, test_loader=test_loader)

        # Collect train loss
        train_loss.append(loss_)

        # Test current SWAG model (if D hat has rank K)
        if len(model.D_hat) == model.K:
            model.theta_swa = theta1.clone().detach()
            model.sigma_vec = theta2 - theta1 ** 2
            p_yxw, p_yx, acc, loss = monte_carlo_bma(model=model, S=model.S, Xtest=Xtest, ytest=ytest, C=model.output_dim)

            # collect test loss and accuracy
            test_loss.append(sum(loss) / len(loss))
            all_acc.append(sum(acc) / len(acc))

        else:
            test_loss.append(0)
            all_acc.append(0)

    # Update class values
    model.theta_swa = theta1.clone().detach()
    model.sigma_vec = theta2 - theta1 ** 2


    # Final plots
    if which_data == 'two_moons':
        decision_path = "SWAG/Decision_K" + str(K) + '_c' + str(c) + "_batch" + str(batch_size) + "_epochs" + str(
            num_epochs) + "_lr" + str(learning_rate) + ".jpg"
        plot_decision_boundary(model, test_loader, S=S,
                               title="Final SWAG with test points", predict_func='stochastic',
                               save_image_path=decision_path)

    loss_path = "SWAG/Loss_accu_MNIST_K" + str(K) + '_c' + str(c) + "_batch" + str(batch_size) + "_epochs" + str(
        num_epochs) + "_lr" + str(learning_rate) + ".jpg"
    plot_acc_and_loss(testloss=test_loss, trainloss=train_loss, accuracy=all_acc, save_path=loss_path)



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
    run = 'deterministic'
    which_data = 'mnist'

    # Parameters
    batch_size = 256
    num_epochs = 250  # må ikke være mindre end K for SWAG
    hidden_dim = 1024
    learning_rate = 0.001
    l2_param = 0.001
    momentum = 0.9

    net_path = 'models/MNIsT/model_1024_drop.pth'
    opti_path = 'Optimizers/MNIST/Opti_1024_drop.pth'

    # net_path = 'models/MNIST/NN_100.pth'
    # opti_path = 'Optimizers/MNIST/Opti_100.pth'

    # Parameters for deterministic
    load_pretrained = False
    save_model = True

    # Parameters for SWAG
    c = 1  # dhat update freq
    K = 10  # dim of dhat
    S = 5  # number of settings bma

    # Load data
    Dataset = LoadDataSet(which_data)
    Xtrain, ytrain, train_loader = Dataset.load_training_data(batch_size=batch_size, noise=0.3)
    Xtest, ytest, test_loader = Dataset.load_test_data(batch_size=batch_size, noise=0.3)

    data = (Xtrain, ytrain, Xtest, ytest)
    loaders = (train_loader, test_loader)

    # Define in- and output dims
    input_dim, output_dim = settings(which_data)

    # Run deterministic net
    if run == 'deterministic':
        main_deterministic(batch_size=batch_size, num_epochs=num_epochs, hidden_dim=hidden_dim,
                           input_dim=input_dim, output_dim=output_dim, learning_rate=learning_rate,
                           net_path=net_path, opti_path=opti_path,
                           data=data, loaders=loaders, load_pretrained=load_pretrained, save_model=save_model,
                           momentum=momentum, l2=l2_param)

    # Run SWAG
    elif run == 'swag':
        main_swag(batch_size=batch_size, num_epochs=num_epochs, hidden_dim=hidden_dim, learning_rate=learning_rate,
                  c=c, K=K, S=S, data=data, loaders=loaders, net_path=net_path, opti_path=opti_path,
                  l2_param=l2_param, input_dim=input_dim, output_dim=output_dim)

    """
    batch_size = 16 # [8, 16, 32, 64, 128] 8 eller 16!!
    num_epochs = 1000
    input_dim = 2
    hidden_dim = 50
    output_dim = 2
    learning_rate = 1.3
    noise = 0.3    """
