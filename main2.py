import torch
import torch.nn as nn
from deterministic import Deterministic_net
from data import DataLoaderInput, LoadDataSet
from utils import plot_decision_boundary, plot_acc_and_loss, compare_parameter_loss_plot
from NN import Net
from SWAG import Swag
from BMA import monte_carlo_bma
import json
from KFAC import KFAC


def settings(data):
    if data == 'two_moons':
        input_dim = 2
        output_dim = 2

    if data == 'mnist':
        input_dim = 784
        output_dim = 10

    return input_dim, output_dim


def train_deterministic(which_data: str, net_path: str, opti_path: str, savePATH: str, param_space: dict, save_net=True,
                        pretrainer=False):
    # dump parameters to json
    with open('params/' + 'savePATH' + '.json', 'w') as fp:
        json.dump(param_space_SWAG, fp)

    if pretrainer:
        param_space['n_epochs'] = 3
        net_path = net_path + 'pretrained.pth'
        opti_path = opti_path + 'pretrained.pth'
    else:
        net_path = net_path + '_epochs' + str(param_space['n_epochs']) + '_batchsize' + str(
            param_space['train_batch_size']) + '.pth'
        opti_path = opti_path + '_epochs' + str(param_space['n_epochs']) + '_batchsize' + str(
            param_space['train_batch_size']) + '.pth'

    # set seed
    torch.manual_seed(param_space['seed'])

    # get dims
    input_dim, output_dim = settings(which_data)

    # Load data
    Dataset = LoadDataSet(which_data)

    if pretrainer and which_data == 'two_moons':
        Xtrain, ytrain, train_loader = Dataset.load_training_data(batch_size=param_space['train_batch_size'], noise=0.3,
                                                                  n_samples=1000)
        Xtest, ytest, test_loader = Dataset.load_test_data(batch_size=param_space['test_batch_size'], noise=0.3,
                                                           n_samples=1000)
    else:
        Xtrain, ytrain, train_loader = Dataset.load_training_data(batch_size=param_space['train_batch_size'], noise=0.3)
        Xtest, ytest, test_loader = Dataset.load_test_data(batch_size=param_space['test_batch_size'], noise=0.3)

    # Define model, optimizer, criterion
    model = Deterministic_net(input_dim, param_space['hidden_dim'], output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=param_space['lr'],
                                weight_decay=param_space['l2'],
                                momentum=param_space['momentum'],
                                nesterov=False)

    # Train model
    if pretrainer:
        print("Deterministic model is (pre)training..")
    else:
        print("Deterministic model is training..")

    test_loss, train_loss, acc = [], [], []
    for epoch in range(param_space['n_epochs']):

        # One epoch of training
        optimizer, criterion, train_loss_ = model.train_net(train_loader, optimizer, criterion,
                                                            save_net=save_net, net_path=net_path, opti_path=opti_path)
        # Test
        accuracy, tst_loss, _ = model.test_net(test_loader=test_loader, criterion=criterion,
                                               freq=30, epoch=epoch)
        # Collect loss
        train_loss.append(float(train_loss_))
        test_loss.append(float(tst_loss.detach().numpy()))
        acc.append(accuracy)

        if epoch % 1 == 0:
            print('Iteration: {}. Test Loss: {}. Train Loss: {}. Accuracy: {}'.format(epoch, tst_loss.item(),
                                                                                      train_loss_, accuracy))
    # Final plots
    if pretrainer:

        if which_data == 'two_moons':
            decision_path = "Plots/" + str(which_data) + "/SWAG/Pretrained_Decision_hidden" + savePATH + ".jpg"

            plot_decision_boundary(model, dataloader=test_loader, S=20,
                                   title="Decision boundary with test points on trained model",
                                   save_image_path=decision_path)

        loss_path = "Plots/" + str(which_data) + "/SWAG/Pretrained_Loss_accu_" + savePATH + ".jpg"
        plot_acc_and_loss(testloss=test_loss, trainloss=train_loss, accuracy=acc, save_path=loss_path)

    else:
        decision_path = "Plots/" + str(which_data) + "/Deterministic/Decision_" + savePATH + ".jpg"

        plot_decision_boundary(model, dataloader=test_loader, S=20,
                               title="Decision boundary with test points on trained model",
                               save_image_path=decision_path)

    loss_path = "Plots/" + str(which_data) + "/Deterministic/Loss_accu_" + savePATH + ".jpg"
    plot_acc_and_loss(testloss=test_loss, trainloss=train_loss, accuracy=acc, save_path=loss_path)


def train_swag(which_data: str, net_path: str, opti_path: str, param_space: dict, save_net=True):
    # Define paths
    net_path = net_path + 'pretrained.pth'
    opti_path = opti_path + 'pretrained.pth'

    # set seed
    torch.manual_seed(param_space['seed'])

    # Define in- and output dims
    input_dim, output_dim = settings(which_data)

    # Load data
    Dataset = LoadDataSet(which_data)
    if which_data == "two_moons":
        Xtrain, ytrain, train_loader = Dataset.load_training_data(batch_size=param_space['train_batch_size'], noise=0.2,
                                                                  n_samples=1000)
        Xtest, ytest, test_loader = Dataset.load_test_data(batch_size=param_space['test_batch_size'], noise=0.2,
                                                           n_samples=100)
    else:
        Xtrain, ytrain, train_loader = Dataset.load_training_data(batch_size=param_space['train_batch_size'], noise=0.2)
        Xtest, ytest, test_loader = Dataset.load_test_data(batch_size=param_space['test_batch_size'], noise=0.2)

    # Define model, optimizer, criterion
    criterion = nn.CrossEntropyLoss()
    model = Swag(input_dim=input_dim, hidden_dim=param_space['hidden_dim'], output_dim=output_dim,
                 K=param_space['K'], c=param_space['c'], S=param_space['S'],
                 criterion=criterion, num_epochs=param_space['n_epochs'],
                 learning_rate=param_space['lr'], l2_param=param_space['l2'])

    optimizer = torch.optim.SGD(model.parameters(), lr=param_space['lr'])

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

    print("Swag is training..")
    # Run T epoch of training and testing
    for i in range(param_space['n_epochs']):

        # Train swag
        n, theta1, theta2, loss_ = model.train_swag(epoch=i, n=n, theta1=theta1, theta2=theta2,
                                                    train_loader=train_loader, test_loader=test_loader)

        # Collect train loss
        train_loss.append(loss_)

        # Test current SWAG model (if D hat has rank K)
        # Every test is performed with new randomly sampled models
        if len(model.D_hat) == model.K:
            model.theta_swa = theta1.clone().detach()
            model.sigma_vec = theta2 - theta1 ** 2
            p_yxw, p_yx, loss, acc = monte_carlo_bma(model=model, S=model.S, Xtest=Xtest, ytest=ytest, C=output_dim)

            # collect test loss and accuracy
            test_loss.append(sum(loss) / len(loss))
            all_acc.append(acc)

        else:
            test_loss.append(0)
            all_acc.append(0)

        if i % 1 == 0 and len(model.D_hat) == model.K:
            print(
                'Iteration: {}. Train Loss: {}, Test loss: {}, BMA accuracy: {}'.format(i, loss_, sum(loss) / len(loss),
                                                                                        acc))
        else:
            print('Iteration: {}. Train Loss: {}, '.format(i, loss_))

    # Update class values
    model.theta_swa = theta1.clone().detach()
    model.sigma_vec = theta2 - theta1 ** 2

    # Final plots
    if which_data == 'two_moons':
        decision_path = "Plots/" + str(which_data) + "/SWAG/Decision_K" + str(param_space['K']) + '_c' + str(
            param_space['c']) + "_batch" + str(param_space['train_batch_size']) + "_epochs" + str(
            param_space['n_epochs']) + "_lr" + str(param_space['lr']) + ".jpg"
        plot_decision_boundary(model, test_loader, S=param_space['S'],
                               title="Final SWAG with test points", predict_func='stochastic',
                               save_image_path=decision_path)

    loss_path = "Plots/" + str(which_data) + "/SWAG/Loss_accu_K" + str(param_space['K']) + '_c' + str(
        param_space['c']) + "_batch" + str(param_space['train_batch_size']) + "_epochs" + str(
        param_space['n_epochs']) + "_lr" + str(param_space['lr']) + ".jpg"
    plot_acc_and_loss(testloss=test_loss, trainloss=train_loss, accuracy=all_acc, save_path=loss_path)


def KFAC_approx(data: str, net_path, opti_path):
    """
    This method computes a KFAC approximation of p(w|D)

    :return:
    """
    param_space = {'device': "cuda" if torch.cuda.is_available() else "cpu",
                   'batch_size': 1,
                   'n_epochs': 1,
                   'seed': 0,
                   'hidden_dim': 10,
                   'save_model': False,
                   'lr': 0.001,
                   'l2': 0.04,
                   'momentum': 0.9,
                   'S': 10,
                   'C': 2,
                   'tau': 0.8,
                   'noise': 0.2}

    # set seed
    torch.manual_seed(param_space['seed'])

    input_dim, output_dim = settings(data)

    # Load data
    Dataset = LoadDataSet(data)
    Xtrain, ytrain, train_loader = Dataset.load_training_data(batch_size = param_space['batch_size'], noise = param_space['noise'], n_samples=2000)
    Xtest, ytest, test_loader = Dataset.load_test_data(batch_size = 1, noise=param_space['noise'], n_samples=1000)

    kfac_model = KFAC(input_dim, param_space['hidden_dim'], output_dim,lr =param_space['lr'] ,
                      momentum = param_space['momentum'],l2_param = param_space['l2'], L = 3)


    # Load pretrained
    kfac_model.load_state_dict(torch.load(net_path))
    optimizer = torch.optim.SGD(kfac_model.parameters(), lr=param_space['lr'])
    optimizer.load_state_dict(torch.load(opti_path))

    for g in optimizer.param_groups:
        g['weight_decay'] = param_space['l2']
        g['lr'] = param_space['lr']
        g['momentum'] = param_space['momentum']


    # collect and regularise
    kfac_model.collect_values(train_loader, optimizer)
    kfac_model.regularize_and_add_prior(tau=param_space['tau'], N=len(Xtrain))

    p_yxw, p_yx, accuracy, all_loss = monte_carlo_bma(kfac_model, Xtest, ytest, S = param_space['S'], C = param_space['C'])

    # Plot bma with KFAC posterior
    if data == 'two_moons':
        plot_decision_boundary(kfac_model, dataloader=test_loader, S=10, title="", predict_func='stochastic',
                                   save_image_path="")


    return p_yxw, p_yx, accuracy, all_loss




if __name__ == '__main__':
    which_data = 'mnist'

    deterministic_net_path = 'models/two_moons/deterministic/model_30'
    deterministic_opti_path = 'Optimizers/two_moons/deterministic/opti_30'

    pretrained_net_path = 'models/mnist/SWAG/model_200'
    pretrained_opti_path = 'Optimizers/mnist/SWAG/opti_200'


    # parameter space for deterministic
    param_space_deterministic = {'device': "cuda" if torch.cuda.is_available() else "cpu",
                                 'train_batch_size': 64,
                                 'test_batch_size': 1000,
                                 'n_epochs': 10,
                                 'seed': 0,
                                 'hidden_dim': 200,
                                 'save_model': False,
                                 'lr': 0.001,
                                 'l2': 0.001,
                                 'momentum': 0.3}

    # parameter space for SWAG

    param_space_SWAG = {'device': "cuda" if torch.cuda.is_available() else "cpu",
                        'train_batch_size': 128,
                        'test_batch_size': 1000,
                        'n_epochs': 100,
                        'seed': 0,
                        'hidden_dim': 200,
                        'save_model': False,
                        'lr': 0.01,
                        'l2': 0.001,
                        'momentum': 0.001,
                        'c': 1,
                        'K': 20,
                        'S': 20}

    # Train deterministic
    # train_deterministic(which_data = which_data, net_path =deterministic_net_path, opti_path = deterministic_opti_path, param_space = param_space_deterministic,save_net = True, pretrainer = False)

    # Pretrain for swag
    train_deterministic(which_data=which_data, net_path=pretrained_net_path, opti_path=pretrained_opti_path,
                        param_space=param_space_deterministic,
                        save_net=True, pretrainer=True)

    # Train swag
    train_swag(which_data=which_data, net_path=pretrained_net_path, opti_path=pretrained_opti_path,
               param_space=param_space_SWAG, save_net=True)

    # Create KFAC posterior
    KFAC_approx(which_data, deterministic_net_path, deterministic_opti_path)

    """
    batch_size = 16 # [8, 16, 32, 64, 128] 8 eller 16!!
    num_epochs = 1000
    input_dim = 2
    hidden_dim = 50
    output_dim = 2
    learning_rate = 1.3
    noise = 0.3    """
