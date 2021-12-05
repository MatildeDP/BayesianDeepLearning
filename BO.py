from deterministic import Deterministic_net
import torch
from sklearn.model_selection import StratifiedKFold
from data import DataLoaderInput, LoadDataSet
import torch.nn as nn
from utils import dump_to_json, json_add_values_to_key, settings
from os.path import exists
import GPyOpt
import numpy as np

def CV_loss(X,y, which_data, params,model, optimizer,K = 2):


    kfold = StratifiedKFold(n_splits=K, shuffle=False)  # get same split every time
    criterion = nn.CrossEntropyLoss()
    # Initialise data containers
    fold_acc = {i:0 for i in range(K)}
    fold_train_loss = {i:0 for i in range(K)}
    fold_test_loss = {i:0 for i in range(K)}

    k = 0  # initialise numner of folders count
    for train_index, test_index in kfold.split(X, y):

        # re-load model and optimizer:
        net_path = 'models/' + which_data.lower() + '/Deterministic/model'
        opti_path = 'Optimizers/' + which_data.lower() + '/Deterministic/opti'
        model.load_state_dict(torch.load(net_path))
        optimizer.load_state_dict(torch.load(opti_path))

        print("CV fold number {} is running...".format(k))

        # Get train and test data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Standardise
        if which_data == 'mnist':
            X_test = (X_test - torch.mean(X_test)) / torch.std(X_test)
            X_train = (X_train - torch.mean(X_train)) / torch.std(X_train)

        # Torch dataloader instance
        train_loader = torch.utils.data.DataLoader(dataset=DataLoaderInput(X_train, y_train),
                                                   batch_size=params['batch_size'],
                                                   shuffle=False)

        test_loader = torch.utils.data.DataLoader(dataset=DataLoaderInput(X_test, y_test),
                                                  batch_size=1000,
                                                  shuffle=False)

        # Epoch loop
        test_loss, train_loss, acc = [], [], []
        for epoch in range(2):
            if epoch % 10 == 0:
                print("Epoch {}".format(epoch))

            # save model in each epoch
            #net_path = 'models/' + which_data.lower() + '/Deterministic/m' + str(m) + '_e' + str(epoch)
            #opti_path = 'Optimizers/' + which_data.lower() + '/Deterministic/m' + str(m) + '_e' + str(epoch)
            #torch.save(model.state_dict(), net_path)
            #torch.save(optimizer.state_dict(), opti_path)

            # One epoch of training
            optimizer, ave_train_loss = model.train_net(train_loader, optimizer, criterion,
                                                        save_net=False)
        # Test
        accuracy, ave_test_loss, _ = model.test_net(test_loader=test_loader, criterion=criterion,
                                                    freq=30)

        # Collect averages acc and loss per fold (overwrites previous models info)
        fold_acc[k] = accuracy.item()
        fold_test_loss[k] = ave_test_loss.item()
        fold_train_loss[k] = ave_train_loss

        print('.......Train loss: {}.      Test loss: {}       Accuracy: {}.'.format(fold_train_loss[k],
                                                                                     fold_test_loss[k],
                                                                                     fold_acc[k]))
        k += 1

    print("Averaged over folds:  Train loss: {}.      Test loss: {}       Accuracy: {}".format(
        sum(fold_train_loss.values()) / len(fold_train_loss),
        sum(fold_test_loss.values()) / len(fold_test_loss),
        sum(fold_acc.values()) / len(fold_acc)))

    print('-' * 100)
    print('-' * 100)

    # dump parameters, test-/train loss, accuracy
    #params['batch_size'] = str(params['batch_size'])  # stringify
    #params['hidden_dim'] = str(params['hidden_dim'])  # stringify
    #model_ = {'params': [params], 'acc': fold_acc, ['train_loss']: fold_train_loss, 'test_loss': [fold_test_loss]}

    #if exists('params_evals/deterministic/model_info_BO.json'):  # dump to existing
    #    json_add_values_to_key('params_evals/deterministic/model_info_BO.json', model_)

   # else:  # create json files if it is the first parameters check
     #   dump_to_json('params_evals/deterministic/model_info_BO.json', model_)

    return sum(fold_test_loss.values())/len(fold_test_loss.values())


def objective_function(x):


    param = x[0]

    # Define data
    which_data = 'mnist'

    # get correct input/output dim dependent on *which_data*
    input_dim, output_dim = settings(which_data)

    print('Parameters: \n Hidden dimension: {} \n l2: {} '
          '\n learning rate: {} \n momentum: {} \n batch size: {}'.format(int(param[4]),
                                                                              param[1], param[0],
                                                                              param[2],
                                                                              int(param[3])))

    param_dict = {'batch_size': int(param[3]), 'hidden_dim':int(param[4]),
              'lr': param[0], 'l2': param[1],
              'momentum': param[2]}

    model = Deterministic_net(input_dim=input_dim, hidden_dim=param_dict['hidden_dim'], output_dim=output_dim)

    optimizer = torch.optim.SGD(model.parameters(), lr=param_dict['lr'],
                                weight_decay=param_dict['l2'],
                                momentum=param_dict['momentum'])

    net_path = 'models/' + which_data.lower() + '/Deterministic/model'
    opti_path = 'Optimizers/' + which_data.lower() + '/Deterministic/opti'
    torch.save(model.state_dict(), net_path)
    torch.save(optimizer.state_dict(), opti_path)




    # load train data
    loss = CV_loss(X, y, which_data, param_dict, model, optimizer)
    return loss


if __name__ == '__main__':

    # load data
    which_data = 'mnist'
    Data = LoadDataSet(which_data)
    X, y, Xval, yval = Data.load_data_for_CV()

    hidden_dim = tuple([torch.arange(10, 350, 1)])

    domain = [{'name': 'lr', 'type': 'continuous', 'domain': (10**-4, 10**-1)}, #torch.logspace(-4, -1, 200)
              {'name': 'l2', 'type': 'continuous', 'domain': (10**-2, 10**-0.1)}, #torch.logspace(-3, -0.1, 200)
              {'name': 'momentum', 'type': 'continuous', 'domain': (10**-2, 10**-0.1)}, #torch.logspace(-2, -0.1, 200)
              {'name': 'batch_size', 'type': 'discrete', 'domain': (8, 16, 32, 64, 128, 256)},
              {'name': 'hidden_dime', 'type': 'discrete', 'domain': range(10, 450)}]
              #{'name': 'n_epochs', 'type': 'discrete', 'domain': range(100, 500)}]

    opt = GPyOpt.methods.BayesianOptimization(f=objective_function,  # function to optimize
                                              domain=domain,  # box-constrains of the problem
                                              acquisition_type='EI',  # Select acquisition function MPI, EI, LCB
                                              )
    # define exploration
    opt.acquisition.exploration_weight = 0.5

    # save report of iterations


    opt.run_optimization(max_iter=1000,report_file = 'BO_reports/report_file.txt', evaluations_file = 'BO_reports/eval_file.txt', models_file='BO_reports/model_file.txt')

    opt.save_evaluations(evaluations_file = 'BO_reports/eval_file.txt') #writing to existing csv file i think
    opt.save_models(models_file = 'BO_reports/model_file.txt')
    opt.save_report('BO_reports/report_file.txt')


    x_best = opt.X[np.argmin(opt.Y)]
    print(x_best)
    print(opt.Y)

    # GP parameter name and values. This is also what we save in model_file
    print(opt.model.get_model_parameters_names())
    print(opt.model.get_model_parameters())

