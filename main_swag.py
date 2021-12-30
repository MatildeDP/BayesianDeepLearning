import torch
import torch.nn as nn
from data import LoadDataSet, settings
import json
import numpy as np
from sklearn.model_selection import train_test_split
from run import run_swag
from Calibration import optimise, optimise_with_temp

if __name__ == '__main__':

    test_each_epoch = True
    which_data = 'two_moons'
    action = 'calibrate' #run_once, optimise, calibrate

    lr_range = torch.arange(1,2,1) # range to optimise swag over
    lr = 0.34 #1#0.006210169289261103 # value when training once
    temp_range = torch.logspace(-0.3,1, 10)
    temp = 1

    # Load BO optimal parameters
    # Opening JSON file
    param_path = 'Deterministic/' + which_data + '/bo/params_evals/model_info_BO.json'
    f = open(param_path)
    data = json.load(f)
    N = len(data['key'])
    all_last_testloss = [data['key'][i][str(i)]['evals']['test_loss'][-1] for i in range(N)]
    argmin = np.argmin(all_last_testloss)
    opti_params = data['key'][argmin][str(argmin)]['params']
    # parameter space for SWAG
    param_space = {'device': "cuda" if torch.cuda.is_available() else "cpu",
                            'batch_size': int(opti_params['batch_size']),
                            'test_batch_size': 1000,
                            'n_epochs': 300,
                            'seed': 0,
                            'hidden_dim': int(opti_params['hidden_dim']),
                            'l2': opti_params['l2'],
                            'momentum': opti_params['momentum'],
                            'c': 1,
                            'K': 20,
                            'S': 30}


    # Define path to pretrained network
    pretrained_net_path = 'SWAG/' + which_data + '/pretrained/model_' + opti_params['hidden_dim']
    pretrained_opti_path = 'SWAG/' + which_data + '/pretrained/opti_'+ opti_params['hidden_dim']

    # Define optimal learning rate:




    # Load data
    Data = LoadDataSet(which_data)
    X, y, Xval, yval = Data.load_data_for_CV()
    input_dim, output_dim = settings(which_data)
    if which_data != 'two_moons':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    else:
        X_train, y_train, X_test, y_test = X, y, Xval, yval

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # optimise chosen method
    if action == 'optimise':
        optimise(alpha_range=lr_range, params=param_space, load_net_path=pretrained_net_path,
                load_opti_path=pretrained_opti_path, model_type='swag', criterion=criterion,
                 X_train=X_train, y_train=y_train, X_test_=X_test, y_test_=y_test, which_data=which_data, test_each_epoch = test_each_epoch)

    elif action == 'calibrate':
        optimise_with_temp(temp_range = temp_range, model_type = 'swag', which_data = which_data, X_test = X_test, y_test = y_test,
                           optimal_model_name = '' , hidden_dim = int(opti_params['hidden_dim']), l2 = opti_params['l2'])

    elif action == 'run_once':
        if temp ==1:
        # path to save info, when training swag once (otherwise this is defined inside optimise/calibrate function
            the_info_path = 'SWAG/' + which_data + '/optimal/params_evals/info.json'
            path_to_bma = 'SWAG/' + which_data + '/optimal/bma/'
            decision_path = 'SWAG/' + which_data + '/optimal/plots/decision_boundary.png'
            loss_path = 'SWAG/' + which_data + '/optimal/plots/loss_acc.png'
            path_to_bma_probs = 'SWAG/' + which_data + '/calibration/probs.json'


        else:
            the_info_path = 'SWAG/' + which_data + '/optimal/params_evals/info_temp.json'
            path_to_bma = 'SWAG/' + which_data + '/optimal/bma/_temp'
            decision_path = 'SWAG/' + which_data + '/optimal/plots/decision_boundary_temp.png'
            loss_path = 'SWAG/' + which_data + '/optimal/plots/loss_acc_temp.png'
            path_to_bma_probs = 'SWAG/' + which_data + '/optimal/probs_temp_' +  str(temp)+'.json'



        run_swag(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                 criterion = criterion, lr = lr , idx_models = 0,
                 load_net_path = pretrained_net_path , load_opti_path = pretrained_opti_path, info_PATH = the_info_path,
                 params = param_space, which_data = which_data,  temp = temp,
                 path_to_bma = path_to_bma, decision_path = decision_path, loss_path = loss_path, test_each_epoch=True,
                 save_probs_path=path_to_bma_probs, count = 0)


