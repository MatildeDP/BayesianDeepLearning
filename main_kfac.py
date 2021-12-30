from run import run_kfac
import torch
import torch.nn as nn
import json
import numpy as np
from data import  LoadDataSet, settings
from sklearn.model_selection import train_test_split
from Calibration import optimise, optimise_with_temp

if __name__ == '__main__':

    test_each_epoch = True
    which_data = 'two_moons'
    action = 'calibrate'  # run_once, optimise, calibrate

    tau_range = torch.logspace(-0.3, 1, 200)  # range to optimise swag over
    tau = 20  # value when training once
    temp_range = torch.logspace(-0.3,1, 2)
    temp = 1


    param_path = 'Deterministic/' + which_data + '/bo/params_evals/model_info_BO.json'
    f = open(param_path)
    data = json.load(f)
    N = len(data['key'])
    all_last_testloss = [data['key'][i][str(i)]['evals']['test_loss'][-1] for i in range(N)]
    model_idx = np.argmin(all_last_testloss)
    opti_params = data['key'][model_idx][str(model_idx)]['params']

    params = {'device': "cuda" if torch.cuda.is_available() else "cpu",
                   'batch_size': 1,
                   'n_epochs': 1,
                   'seed': 0,
                   'hidden_dim': int(opti_params['hidden_dim']),
                   'lr': opti_params['lr'],
                   'l2': opti_params['l2'],
                   'momentum': opti_params['momentum'],
                   'S': 30,
                   'c': ''}


    # Loads model trained with optimal parameters
    pretrained_net_path = 'Deterministic/' + which_data + '/optimal/model'
    pretrained_opti_path = 'Deterministic/' + which_data + '/optimal/optimizer'

    # Load data
    Data = LoadDataSet(which_data)
    X, y, Xval, yval = Data.load_data_for_CV()
    input_dim, output_dim = settings(which_data)
    if which_data != 'two_moons':
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    else:
        _, _, X_test, y_test = X, y, Xval, yval

   # _, X_test, _, y_test = train_test_split(X_, y_, test_size=0.2, random_state=0, stratify=y_)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # optimise chosen method
    if action == 'optimise':
        optimise(alpha_range=tau_range, params=params, load_net_path=pretrained_net_path,
                 load_opti_path=pretrained_opti_path, model_type='kfac', criterion=criterion,
                 X_train=0, y_train=0, X_test_=X_test, y_test_=y_test, which_data=which_data,
                 test_each_epoch = test_each_epoch)

    elif action == 'calibrate':
        optimise_with_temp(temp_range=temp_range, model_type='kfac', which_data=which_data, X_test=X_test,
                           y_test=y_test,
                           optimal_model_name='', hidden_dim=int(opti_params['hidden_dim']), l2=opti_params['l2'])

    elif action == 'run_once':
        # paths to save stuff, when training swag once (otherwise this is defined inside optimise/calibrate function
        if temp ==1:
            info_PATH = 'KFAC/' + which_data + '/optimal/params_evals/info.json'
            path_to_bma = 'KFAC/' + which_data + '/optimal/bma/'
            decision_path = 'KFAC/' + which_data + '/optimal/plots/decision_boundary.png'
            loss_path = 'KFAC/' + which_data + '/optimal/plots/loss_acc.png'
            #path_to_bma_probs = 'KFAC/' + which_data + '/calibration/probs/probs.json'
            path_to_bma_probs = ''
        else:
            info_PATH = 'KFAC/' + which_data + '/optimal/params_evals/info_temp.json'
            path_to_bma = 'KFAC/' + which_data + '/optimal/bma/temp'
            decision_path = 'KFAC/' + which_data + '/optimal/plots/decision_boundary_temp.png'
            loss_path = 'KFAC/' + which_data + '/optimal/plots/loss_acc.png_temp'
            #path_to_bma_probs = 'KFAC/' + which_data + '/calibration/probs/probs_temp_' + str(
            #    temp) + 'json'
            path_to_bma_probs = ''

        run_kfac(tau = tau, params = params, load_net_path = pretrained_net_path, load_opti_path = pretrained_opti_path,
                 X_train = 0, y_train = 0, X_test = X_test, y_test = y_test, criterion = criterion,
                 which_data= which_data, loss_path = loss_path, decision_path = decision_path, info_PATH = info_PATH,
                 path_to_bma = path_to_bma, idx_model = 0, N = X_test.shape[0], temp = temp,
                 test_each_epoch = test_each_epoch, save_probs_path=path_to_bma_probs, count = 0)
