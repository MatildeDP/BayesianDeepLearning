import torch
import torch.nn as nn
from data import LoadDataSet, settings
from sklearn.model_selection import train_test_split
from run import train_deterministic
from Calibration import optimise_with_temp
import json
import numpy as np
from time import time

if __name__ == '__main__':



    test_each_epoch = True
    which_data = 'two_moons' # two_moons, mnist, abelone, cancer, emnist
    temp = 1
    action = 'calibrate_ensample' # run_once, calibrate, pretrain_for_swag ensample, calibrate_ensample
    param_path = 'Deterministic/'+which_data+'/bo/params_evals/model_info_BO.json'

    temp_range = torch.logspace(-0.3,1, 5)#torch.arange(0.01,5,0.1)

    # Load BO optimal parameters
    # Opening JSON file
    f = open(param_path)
    data = json.load(f)
    N = len(data['key'])
    all_last_testloss = [data['key'][i][str(i)]['evals']['test_loss'][-1] for i in range(N)]
    model_idx = np.argmin(all_last_testloss)
    opti_params = data['key'][model_idx][str(model_idx)]['params']

    param_space = {'batch_size': int(opti_params['batch_size']),
                   'test_batch_size': 1000,
                   'n_epochs': 300,
                   'seed': 0,
                   'hidden_dim': 230,
                   'lr': opti_params['lr'],
                   'l2': opti_params['l2'],
                   'momentum':opti_params['momentum']}

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

    # path to save model at
    if action == 'pretrain_for_swag':
        net_path = 'SWAG/' + which_data + '/pretrained/model_' + str(param_space['hidden_dim'])
        opti_path ='SWAG/' + which_data + '/pretrained/opti_' + str(param_space['hidden_dim'])
        infoPATH = 'SWAG/' + which_data + '/pretrained/info_' + str(param_space['hidden_dim']) + '.json'
        decision_path = 'SWAG/' + which_data + '/pretrained/decision_boundary_' + str(param_space['hidden_dim']) + '.png'
        loss_path ='SWAG/' + which_data + '/pretrained/loss_acc_' + str(param_space['hidden_dim']) + '.png'
        param_space['n_epochs'] = 2

        # Train deterministic
        train_deterministic(which_data = which_data, net_path = net_path, opti_path = opti_path, infoPATH = infoPATH,
                            param_space = param_space, temp = temp, X_train = X_train, y_train = y_train,
                            X_test = X_test, y_test = y_test, decision_path = decision_path, loss_path = loss_path, idx_model = 0,
                            test_each_epoch = test_each_epoch)

    elif action == 'calibrate':
        optimise_with_temp(temp_range, model_type = 'deterministic', which_data = which_data, X_test = X_test, y_test= y_test,
                           optimal_model_name = 'm_'+str(model_idx), hidden_dim =  int(opti_params['hidden_dim']), l2 = opti_params['l2'])



    # save model with optimal parameters
    elif action == 'run_once':
        if temp == 1:
            net_path = 'Deterministic/'+which_data+'/optimal/model' + opti_params['hidden_dim']
            opti_path = 'Deterministic/'+which_data+'/optimal/optimizer' + opti_params['hidden_dim']
            infoPATH = 'Deterministic/' + which_data + '/optimal/info_.json'
            decision_path = 'Deterministic/' + which_data + '/optimal/decision_boundary_.png'
            loss_path ='Deterministic/' + which_data + '/optimal/loss_acc_.png'
            # Train deterministic
            train_deterministic(which_data=which_data, net_path=net_path, opti_path=opti_path, infoPATH=infoPATH,
                                param_space=param_space, temp=temp, X_train=X_train, y_train=y_train,
                                X_test=X_test, y_test=y_test, decision_path=decision_path, loss_path=loss_path,
                                idx_model = 0, test_each_epoch= test_each_epoch)
        else:
            net_path = 'Deterministic/'+which_data+'/optimal/model_temp' + opti_params['hidden_dim']
            opti_path = 'Deterministic/'+which_data+'/optimal/optimizer_temp' + opti_params['hidden_dim']
            infoPATH = 'Deterministic/' + which_data + '/optimal/info_temp.json'
            decision_path = 'Deterministic/' + which_data + '/optimal/decision_boundary_temp.png'
            loss_path ='Deterministic/' + which_data + '/optimal/loss_acc_temp.png'

            # Train deterministic
            train_deterministic(which_data=which_data, net_path=net_path, opti_path=opti_path, infoPATH=infoPATH,
                                param_space=param_space, temp=temp, X_train=X_train, y_train=y_train,
                                X_test=X_test, y_test=y_test, decision_path=decision_path, loss_path=loss_path,
                                idx_model = 0, test_each_epoch = test_each_epoch)

    elif action == 'calibrate_ensample':
        optimise_with_temp(temp_range, model_type = 'ensample', which_data = which_data, X_test = X_test, y_test= y_test,
                           optimal_model_name = '', hidden_dim =  int(opti_params['hidden_dim']), l2 = opti_params['l2'])



    elif action == 'ensample':

        for i in range(5):
            param_space['seed'] = i+10
            net_path = 'Deterministic/'+which_data+'/ensample/model_' + str(i)
            opti_path = 'Deterministic/'+which_data+'/ensample/optimizer'  + str(i)
            infoPATH = 'Deterministic/' + which_data + '/ensample/info_.json' + str(i)
            decision_path = 'Deterministic/' + which_data + '/ensample/decision_boundary_' +str(i)+'.png'
            loss_path ='Deterministic/' + which_data + '/ensample/loss_acc_' +str(i) +'.png'
            # Train deterministic
            train_deterministic(which_data=which_data, net_path=net_path, opti_path=opti_path, infoPATH=infoPATH,
                                param_space=param_space, temp=temp, X_train=X_train, y_train=y_train,
                                X_test=X_test, y_test=y_test, decision_path=decision_path, loss_path=loss_path,
                                idx_model = 0, test_each_epoch= test_each_epoch)

















