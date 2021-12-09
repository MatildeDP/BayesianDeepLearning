import torch
import torch.nn as nn
from data import LoadDataSet, settings
from sklearn.model_selection import train_test_split
from run import train_deterministic
from Calibration import optimise_with_temp

if __name__ == '__main__':

    test_each_epoch = True
    which_data = 'two_moons'
    pretrain_for_swag = False
    temp = 1
    action = 'calibrate' # run_once, calibrate

    temp_range = torch.logspace(-4,0.7,10)

    # parameter space for deterministic
    if which_data == 'mnist':
        # TODO: Mangler stadig de helt optimale værdier
        optimal_params_PATH = 'Deterministic/mnist/optimal' # + filename
        param_space = {'batch_size': 32,
                                     'test_batch_size': 1000,
                                     'n_epochs': 500,
                                     'seed': 0,
                                     'hidden_dim': 126,
                                     'lr': 0.00579715471546691,
                                     'l2': 0.0006577587247703735,
                                     'momentum': 0.6105694643236608}

    elif which_data == 'two_moons':
        # TODO: Mangler stadig de helt optimale værdier
        optimal_params_PATH = 'Deterministic/two_moons/optimal'  # + filename
        param_space = {'batch_size': 32,
                       'test_batch_size': 1000,
                       'n_epochs': 200,
                       'seed': 0,
                       'hidden_dim': 126,
                       'lr': 0.00579715471546691,
                       'l2': 0.0006577587247703735,
                       'momentum': 0.6105694643236608}



    # Load data
    Data = LoadDataSet(which_data)
    X, y, Xval, yval = Data.load_data_for_CV()
    input_dim, output_dim = settings(which_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # path to save model at
    if pretrain_for_swag:
        net_path = 'SWAG/' + which_data + '/pretrained/model_' + str(param_space['hidden_dim'])
        opti_path ='SWAG/' + which_data + '/pretrained/opti_' + str(param_space['hidden_dim'])
        infoPATH = 'SWAG/' + which_data + '/pretrained/info_' + str(param_space['hidden_dim']) + '.json'
        decision_path = 'SWAG/' + which_data + '/pretrained/decision_boundary_' + str(param_space['hidden_dim']) + '.png'
        loss_path ='SWAG/' + which_data + '/pretrained/loss_acc_' + str(param_space['hidden_dim']) + '.png'
        param_space['n_epochs'] = 5

        # Train deterministic
        train_deterministic(which_data = which_data, net_path = net_path, opti_path = opti_path, infoPATH = infoPATH,
                            param_space = param_space, temp = temp, X_train = X_train, y_train = y_train,
                            X_test = X_test, y_test = y_test, decision_path = decision_path, loss_path = loss_path, idx_model = 0)

    elif action == 'calibrate':
        optimise_with_temp(alpha_range = None,params = param_space, temp_range = temp_range,
                           load_net_path = '', load_opti_path = '', model_type = 'deterministic',
                           which_data = which_data,
                           X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                           criterion = criterion, test_each_epoch = test_each_epoch)



    # save model with optimal parameters
    elif action == 'run_once':
        if temp == 1:
            net_path = 'Deterministic/'+which_data+'/optimal/model'
            opti_path = 'Deterministic/'+which_data+'/optimal/optimizer'
            infoPATH = 'Deterministic/' + which_data + '/optimal/info_.json'
            decision_path = 'Deterministic/' + which_data + '/optimal/decision_boundary_.png'
            loss_path ='Deterministic/' + which_data + '/optimal/loss_acc_.png'
            # Train deterministic
            train_deterministic(which_data=which_data, net_path=net_path, opti_path=opti_path, infoPATH=infoPATH,
                                param_space=param_space, temp=temp, X_train=X_train, y_train=y_train,
                                X_test=X_test, y_test=y_test, decision_path=decision_path, loss_path=loss_path,
                                idx_model = 0, test_each_epoch= test_each_epoch)
        else:
            net_path = 'Deterministic/'+which_data+'/optimal/model_temp'
            opti_path = 'Deterministic/'+which_data+'/optimal/optimizer_temp'
            infoPATH = 'Deterministic/' + which_data + '/optimal/info_temp.json'
            decision_path = 'Deterministic/' + which_data + '/optimal/decision_boundary_temp.png'
            loss_path ='Deterministic/' + which_data + '/optimal/loss_acc_temp.png'

            # Train deterministic
            train_deterministic(which_data=which_data, net_path=net_path, opti_path=opti_path, infoPATH=infoPATH,
                                param_space=param_space, temp=temp, X_train=X_train, y_train=y_train,
                                X_test=X_test, y_test=y_test, decision_path=decision_path, loss_path=loss_path,
                                idx_model = 0, test_each_epoch = test_each_epoch)








