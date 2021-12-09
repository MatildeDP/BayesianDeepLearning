import torch
import torch.nn as nn
from data import LoadDataSet, settings

from sklearn.model_selection import train_test_split
from run import run_swag
from Calibration import optimise, optimise_with_temp

if __name__ == '__main__':

    test_each_epoch = True
    which_data = 'two_moons'
    action = 'optimise' #run_once, optimise, calibrate

    lr_range = torch.logspace(-2, 0, 10) # range to optimise swag over
    lr = 0.01  # value when training once
    temp_range = torch.logspace(-2, 0, 10)
    temp = 2


    # parameter space for SWAG
    param_space = {'device': "cuda" if torch.cuda.is_available() else "cpu",
                        'batch_size': 32,
                        'test_batch_size': 1000,
                        'n_epochs': 200,
                        'seed': 0,
                        'hidden_dim': 200,
                        'save_model': False,
                        'l2': 0.001,
                        'momentum': 0.001,
                        'c': 1,
                        'K': 20,
                        'S': 20}


    # Define path to pretrained network
    pretrained_net_path = 'SWAG/' + which_data + '/pretrained/model_200'
    pretrained_opti_path = 'SWAG/' + which_data + '/pretrained/opti_200'


    # Load data
    Data = LoadDataSet(which_data)
    X, y, Xval, yval = Data.load_data_for_CV()
    input_dim, output_dim = settings(which_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # optimise chosen method
    if action == 'optimise':
        optimise(alpha_range=lr_range, params=param_space, load_net_path=pretrained_net_path,
                load_opti_path=pretrained_opti_path, model_type='swag', criterion=criterion,
                 X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, which_data=which_data, test_each_epoch = test_each_epoch)

    elif action == 'calibrate':
        optimise_with_temp(alpha_range = lr_range, params = param_space, temp_range = temp_range,
                           load_net_path =pretrained_net_path, load_opti_path = pretrained_opti_path,
                           model_type = 'swag', which_data = which_data,
                           X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, criterion = criterion)

    elif action == 'run_once':
        if temp ==1:
        # path to save info, when training swag once (otherwise this is defined inside optimise/calibrate function
            the_info_path = 'SWAG/' + which_data + '/optimal/params_evals/info.json'
            path_to_bma = 'SWAG/' + which_data + '/optimal/bma/'
            decision_path = 'SWAG/' + which_data + '/optimal/plots/decision_boundary.png'
            loss_path = 'SWAG/' + which_data + '/optimal/plots/loss_acc.png'

        else:
            the_info_path = 'SWAG/' + which_data + '/optimal/params_evals/info_temp.json'
            path_to_bma = 'SWAG/' + which_data + '/optimal/bma/_temp'
            decision_path = 'SWAG/' + which_data + '/optimal/plots/decision_boundary_temp.png'
            loss_path = 'SWAG/' + which_data + '/optimal/plots/loss_acc_temp.png'


        run_swag(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                 criterion = criterion, lr = lr , idx_models = 0,
                 load_net_path = pretrained_net_path , load_opti_path = pretrained_opti_path, info_PATH = the_info_path,
                 params = param_space, which_data = which_data,  temp = temp,
                 path_to_bma = path_to_bma, decision_path = decision_path, loss_path = loss_path, test_each_epoch=True)


