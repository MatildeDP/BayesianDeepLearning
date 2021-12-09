from data import LoadDataSet, settings
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from run import run_swag, run_kfac, train_deterministic


## Optimising stochastic models

# load data
def optimise_with_temp(alpha_range, params: dict, temp_range, load_net_path: str, load_opti_path: str, model_type: str, which_data: str, X_train, y_train, X_test, y_test, criterion, test_each_epoch):
    """
    :param alpha_range:
    :param params:
    :param temp_range:
    :param load_net_path:
    :param load_opti_path:
    :param model_type:
    :param which_data:
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param criterion:
    :return:
    """

    # Grid search loop
    idx_models = 0
    if model_type == 'swag':
        for alpha in enumerate(range(len(alpha_range))):
            # Calibrate with temperature
            for temp in temp_range:
                path_to_bma = 'SWAG/' + which_data + '/calibration/bma/temp_' + str(temp)
                info_PATH = 'SWAG/' + which_data + '/calibration/params_evals/info.json'

                # paths to plots
                decision_path = 'SWAG/' + which_data + '/calibration/'+ str(idx_models) + "_decision_boundary.jpg"
                loss_path = 'SWAG/' + which_data + '/calibration/' + str(idx_models) + "_loss_acc.jpg" # loss plots is created if we test every epoch

                run_swag(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                          criterion=criterion, lr=alpha, idx_models=idx_models,
                          load_net_path=load_net_path, load_opti_path=load_opti_path, info_PATH=info_PATH,
                          params=params, which_data=which_data, temp=temp, path_to_bma = path_to_bma,
                         decision_path = decision_path, loss_path = loss_path, test_each_epoch = test_each_epoch )

                idx_models += 1

    elif model_type == 'kfac':
        idx_models = 0
        for alpha in enumerate(alpha_range):
            # Calibrate with temperature
            for temp in temp_range:
                path_to_bma = 'KFAC/' + which_data + '/gridsearch/bma/'
                info_PATH = 'KFAC/' + which_data + '/gridsearch/params_evals/info.json'

                # paths to plots
                decision_path = 'KFAC/' + which_data + '/gridsearch/plots/' + str(idx_models) + '_decision_boundary.jpg'
                loss_path = 'KFAC/' + which_data + '/gridsearch/plots/' + str(
                    idx_models) + "_loss_acc.jpg"  # loss plots is created if we test every epoch
                N = X_train.shape[0]

                run_kfac(tau=alpha, params=params, X_train=X_train, y_train=y_train,
                         X_test=X_test, y_test=y_test, criterion=criterion, which_data=which_data,
                         load_net_path=load_net_path, load_opti_path=load_opti_path,
                         loss_path=loss_path, decision_path=decision_path, path_to_bma=path_to_bma,
                         idx_model=idx_models, temp = temp,
                         test_each_epoch=test_each_epoch, N=N, info_PATH=info_PATH)
                idx_models += 1

    if model_type == 'deterministic':
        for idx_models, temp in enumerate(temp_range):
            print('temperature: {}'.format(temp))
            model_path = 'Deterministic/' + which_data + '/calibrating/models/model_' + str(idx_models)
            opti_path = 'Deterministic/' + which_data + '/calibrating/opti/opti_' + str(idx_models)
            info_PATH = 'Deterministic/' + which_data + '/calibrating/params_evals/info.json'

            # paths to plots
            decision_path = 'Deterministic/' + which_data + '/calibrating/plots/' + str(idx_models) + '_decision_boundary.jpg'
            loss_path = 'Deterministic/' + which_data + '/calibrating/plots/' + str(idx_models) + "_loss_acc.jpg"  # loss plots is created if we test every epoch

            train_deterministic(which_data = which_data, net_path = model_path, opti_path = opti_path, infoPATH = info_PATH,
                                param_space = params, temp = temp, X_train = X_train, y_train = y_train, X_test = X_test,
                                y_test = y_test, decision_path = decision_path, loss_path = loss_path, idx_model = idx_models, test_each_epoch = test_each_epoch)





def optimise(alpha_range, params: dict, load_net_path: str, load_opti_path: str, model_type: str, X_train, y_train, X_test, y_test, criterion, which_data, test_each_epoch):
    """

    :param alpha_range:
    :param params:
    :param load_net_path:
    :param load_opti_path:
    :param model_type:
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param criterion:
    :param which_data:
    :return:
    """


    if model_type == 'swag':

        # Grid search loop
        for idx_models, alpha in enumerate(alpha_range):
            path_to_bma = 'SWAG/'+which_data+'/gridsearch/bma/'
            info_PATH = 'SWAG/'+which_data+'/gridsearch/params_evals/info.json'

            # paths to plots
            decision_path = 'SWAG/' + which_data + '/gridsearch/plots/' + str(idx_models) + '_decision_boundary.jpg'
            loss_path = 'SWAG/' + which_data + '/gridsearch/plots/' + str(idx_models) + "_loss_acc.jpg"  # loss plots is created if we test every epoch

            run_swag(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                      criterion = criterion, lr = alpha, idx_models = idx_models,
                      load_net_path = load_net_path, load_opti_path = load_opti_path, info_PATH = info_PATH,
                      params = params, which_data = which_data, temp = 1, path_to_bma = path_to_bma,
                     decision_path = decision_path, loss_path = loss_path, test_each_epoch = test_each_epoch)

    if model_type == 'kfac':

        for idx_models, alpha in enumerate(alpha_range):
            path_to_bma = 'KFAC/'+which_data+'/gridsearch/bma/'
            info_PATH = 'KFAC/'+which_data+'/gridsearch/params_evals/info.json'

            # paths to plots
            decision_path = 'KFAC/' + which_data + '/gridsearch/plots/' + str(idx_models) + '_decision_boundary.jpg'
            loss_path = 'KFAC/' + which_data + '/gridsearch/plots/' + str(idx_models) + "_loss_acc.jpg"  # loss plots is created if we test every epoch
            N = X_train.shape[0]

            run_kfac(tau = alpha, params = params, X_train = X_train, y_train = y_train,
                     X_test = X_test, y_test = y_test, criterion = criterion, which_data = which_data,
                    load_net_path = load_net_path, load_opti_path = load_opti_path,
                    loss_path = loss_path, decision_path = decision_path, path_to_bma = path_to_bma, idx_model = idx_models,
                    test_each_epoch = test_each_epoch, temp = 1, N = N, info_PATH = info_PATH)







if __name__ == '__main__':

    method = 'swag'
    which_data = 'two_moons'
    K = 2

    params = {'device': "cuda" if torch.cuda.is_available() else "cpu",
              'batch_size': 32,
              'test_batch_size': 1000,
              'n_epochs': 200,
              'seed': 0,
              'hidden_dim': 200,
              'l2': 0.001,
              'momentum': 0.001,
              'c': 1,
              'K': 20,
              'S': 20}


    alpha_range = torch.logspace(-2, 0, 10)

    # Denine path to pretrained network
    if method == 'swag':
        if which_data == 'mnist':
            pretrained_net_path = 'models/mnist/SWAG/model_200pretrained.pth'
            pretrained_opti_path = 'Optimizers/mnist/SWAG/opti_200pretrained.pth'

        elif which_data == 'two_moons':
            pretrained_net_path = 'pretrained/two_moons/model_200pretrained_for_swag.pth'
            pretrained_opti_path = 'pretrained/two_moons/opti_200pretrained_for_swag.pth'
            pass

    if method == 'kfac':
        if which_data == 'mnist':
            pass

        elif which_data == 'two_moons':
            pass

    # Load data
    Data = LoadDataSet(which_data)
    X, y, Xval, yval = Data.load_data_for_CV()
    input_dim, output_dim = settings(which_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Define loss function
    criterion = nn.CrossEntropyLoss()


    # optimise chosen method
    optimise(alpha_range=alpha_range, params=params, load_net_path=pretrained_net_path,
                  load_opti_path=pretrained_opti_path, model_type = method, criterion = criterion,
             X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, which_data = which_data)


    abc = 123