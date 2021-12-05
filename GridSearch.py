import torch.nn as nn
import torch
from sklearn.model_selection import StratifiedKFold
from data import DataLoaderInput, LoadDataSet
from deterministic import Deterministic_net
import numpy as np
import random
import json
from utils import settings, dump_to_json, dump_to_existing_json


def random_grid_search_CV(momentum_range, lr_range, l2_range, hidden_dim_range, batch_size_range, num_models, n_epochs, K):
    # Define loss function
    criterion = nn.CrossEntropyLoss()
    kfold = StratifiedKFold(n_splits=K, shuffle=False)  # get same split every time

    # Initialise data containers
    fold_acc = {i: {j: 0 for j in range(num_models)} for i in range(K)}
    fold_train_loss = {i: {j: 0 for j in range(num_models)} for i in range(K)}
    fold_test_loss = {i: {j: 0 for j in range(num_models)} for i in range(K)}
    current_best = 0

    # loop through params
    for m in range(num_models):

        params = {'batch_size': random.choice(batch_size_range), 'hidden_dim': random.choice(hidden_dim_range).item(),
                  'lr':random.choice(lr_range).item(), 'l2': random.choice(l2_range).item(), 'momentum': random.choice(momentum_range).item()}

        print('Training model {} with parameters: \n Hidden dimension: {} \n l2: {} '
              '\n learning rate: {} \n momentum: {} \n batch size: {} \n '.format(m, params['hidden_dim'],
                                                                                  params['l2'], params['lr'],
                                                                                  params['momentum'],
                                                                                  params['batch_size']))

        k = 0  # initialise numner of folders count
        for train_index, test_index in kfold.split(X, y):

            print("CV fold number {} is running...".format(k))

            # Get train and test data
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Standardise
            X_test = (X_test - torch.mean(X_test)) / torch.std(X_test)
            X_train = (X_train - torch.mean(X_train)) / torch.std(X_train)

            # Torch dataloader instance
            train_loader = torch.utils.data.DataLoader(dataset=DataLoaderInput(X_train, y_train),
                                                       batch_size=params['batch_size'],
                                                       shuffle=False)

            test_loader = torch.utils.data.DataLoader(dataset=DataLoaderInput(X_test, y_test),
                                                      batch_size=1000,
                                                      shuffle=False)

            # Initialise model and optimizer
            model = Deterministic_net(input_dim=input_dim, hidden_dim=params['hidden_dim'], output_dim=output_dim)

            optimizer = torch.optim.SGD(model.parameters(), lr=params['lr'],
                                        weight_decay=params['l2'],
                                        momentum=params['momentum'])

            # Epoch loop
            for epoch in range(n_epochs):
                if epoch % 10 == 0:
                    print("Epoch {}".format(epoch))

                # save model in each epoch
                net_path = 'models/' + which_data.lower() + '/Deterministic/m' + str(m) + '_e' + str(epoch)
                opti_path = 'Optimizers/' + which_data.lower() + '/Deterministic/m' + str(m) + '_e' + str(epoch)
                torch.save(model.state_dict(), net_path)
                torch.save(optimizer.state_dict(), opti_path)

                # One epoch of training
                optimizer, ave_train_loss = model.train_net(train_loader, optimizer, criterion,
                                                            save_net= False)
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
        if m == 0:  # create json files if model is the first
            params['batch_size'] = str(params['batch_size'])  # stringify
            params['hidden_dim'] = str(params['hidden_dim'])  # stringify
            model_ = {m: {'params': params, 'acc': fold_acc, 'train_loss': fold_train_loss, 'test_loss': fold_test_loss}}

            dump_to_json('params_evals/deterministic/model_info.json', model_)


        else:  # dump to existing
            params['batch_size'] = str(params['batch_size'])  # stringify
            params['hidden_dim'] = str(params['hidden_dim'])  # stringify
            model_ = {m: {'params': params, 'acc': fold_acc, 'train_loss': fold_train_loss, 'test_loss': fold_test_loss}}

            dump_to_existing_json('params_evals/deterministic/model_info.json', model_)

            # dump to existing json

        # Check if test loss is better than previous models

        # temp = sum(fold_test_loss.values())/len(fold_test_loss.values())
        # if temp > current_best:

        # current_best = sum(fold_test_loss.values()) / len(fold_test_loss.values())  # Average test loss over folds
        # params['batch_size'] = str(params['batch_size'])  # stringify
        # params['hidden_dim'] = str(params['hidden_dim'])  # stringify
        # dump_to_json('params_evals/deterministic/models.json', params)
        # dump_to_json('params_evals/deterministic/train_loss.json', train_loss)
        # dump_to_json('params_evals/deterministic/test_loss.json', test_loss)
        # dump_to_json('params_evals/deterministic/acc.json', acc)

        # elif temp == current_best:
        #    print('Two models with equally good test loss have been found.')


if __name__ == '__main__':

    which_data = 'mnist'

    # load data
    Data = LoadDataSet(which_data)
    X, y, Xval, yval = Data.load_data_for_CV()

    input_dim, output_dim = settings(which_data)

    # Define ranges
    batch_size_mnist = [32, 64, 128, 256]
    batch_size_moons = [16, 32, 64, 128]

    hidden_dim_mnist = torch.arange(100, 250, 10) # stepsize 5 to avoid choosing numbers to close to each other
    hidden_dim_two_moons = torch.arange(10, 100, 10)

    if which_data == 'mnist':
        hidden_dim = hidden_dim_mnist
        batch_size = batch_size_mnist

    elif which_data == 'two_moons':
        hidden_dim = hidden_dim_two_moons
        batch_size = batch_size_moons

    lr = torch.logspace(-4, -1, 20)
    l2 = torch.logspace(-3, -0.1, 20)
    momentum = torch.logspace(-2, -0.1, 20)

    num_models = 3
    K = 2  # folds
    n_epochs = 5
    random_grid_search_CV(momentum_range = momentum, lr_range = lr, l2_range = l2, hidden_dim_range = hidden_dim,
                          batch_size_range = batch_size, num_models = num_models, n_epochs = n_epochs, K = K)



