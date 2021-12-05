from SWAG import Swag
from data import LoadDataSet, DataLoaderInput
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from utils import settings, dump_to_json, dump_to_existing_json
from BMA import monte_carlo_bma


## Optimising stochastic models

# load data
def optimise_swag(which_data: str, n_splits: int, alpha_range, params: dict, net_path: str, opti_path: str, T_scale: bool):
    """
    Method chooses parameters based on test loss computed on model trained on n epochs
    :param data: name of dataset
    :param n_splits: number of CV splits
    :param alpha_range: range of parameter to be tuned (lr for swag and tau for kfac)
    :param params: fixed paramters
    :return:
    """

    # Define path to model info
    if T_scale:
        info_PATH = 'params_evals/SWAG/model_info_temperature_scaled.json'
    else:
        info_PATH = 'params_evals/SWAG/model_info.json'


    # Load data
    Data = LoadDataSet(which_data)
    X, y, Xval, yval = Data.load_data_for_CV()
    input_dim, output_dim = settings(which_data)

    # Define splits
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Grid search loop
    for idx_models, alpha in enumerate(range(len(alpha_range))):

        # run extra loop to optimise temperature scale parameter
        if T_scale:
            pass
        else:
            CV_SWAG(input_dim, output_dim, criterion, alpha, idx_models, kfold, X, y, net_path, opti_path, info_PATH, params)



def CV_SWAG(input_dim,output_dim, criterion, lr, idx_models, kfold, X, y, net_path, opti_path, info_PATH, params):
    # Define model with correct learning rate
    model = Swag(input_dim=input_dim, hidden_dim=params['hidden_dim'], output_dim=output_dim,
                 K=params['K'], c=params['c'], S=params['S'],
                 criterion=criterion, num_epochs=params['n_epochs'],
                 learning_rate=lr, l2_param=params['l2'])

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    print('Training model {} with parameters: \n Hidden dimension: {} \n l2: {} '
          '\n learning rate: {} \n momentum: {} \n batch size: {} \n c(update freq): {}. \n K(Dhat rank): {} \n S(# samples in BMA): {}'.format(
        idx_models, params['hidden_dim'],
        params['l2'], lr,
        params['momentum'],
        params['batch_size'], params['c'], params['K'], params['S']))

    # Initialise data containers
    fold_acc = {i: 0 for i in range(K)}
    fold_train_loss = {i: 0 for i in range(K)}
    fold_test_loss = {i: 0 for i in range(K)}


    k = 0
    # CV loop
    for train_index, test_index in kfold.split(X, y):
        print("CV fold number {} is running...".format(k))

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Get train and test data
        if which_data == 'two_moons':
            pass
        else:  # Standardise
            X_test = (X_test - torch.mean(X_test)) / torch.std(X_test)
            X_train = (X_train - torch.mean(X_train)) / torch.std(X_train)

        # Torch dataloader instance
        train_loader = torch.utils.data.DataLoader(dataset=DataLoaderInput(X_train, y_train),
                                                   batch_size=params['batch_size'],
                                                   shuffle=False)

        test_loader = torch.utils.data.DataLoader(dataset=DataLoaderInput(X_test, y_test),
                                                  batch_size=1000,
                                                  shuffle=False)

        # Re define model and optimizer to ensure same start conditions
        model.load_state_dict(torch.load(net_path))
        optimizer.load_state_dict(torch.load(opti_path))

        for g in optimizer.param_groups:
            g['lr'] = lr,
            g['weight_decay'] = params['l2'],
            g['momentum'] = params['momentum']

        ############################################################################################################
        # Train and test
        # Extract initial weights
        theta1, theta2 = model.get_ith_moments()

        # Initialize error and accuracy container
        all_train_loss = []
        n = 0

        print("Swag is training..")
        # Run T epoch of training and testing
        for i in range(params['n_epochs']):
            if i % 10 == 0:
                print("Epoch {}".format(i))

            # Train swag
            n, theta1, theta2, train_Loss = model.train_swag(epoch=i, n=n, theta1=theta1, theta2=theta2,
                                                             train_loader=train_loader, test_loader=test_loader)

        # Update moments after training
        model.theta_swa = theta1.clone().detach()
        model.sigma_vec = theta2 - theta1 ** 2

        # Test on trained model
        p_yxw, p_yx, test_loss, acc = monte_carlo_bma(model=model, S=model.S, Xtest=X_test, ytest=y_test,
                                                      C=output_dim)

        # Collect average acc and loss per fold (overwrites previous model info)
        fold_acc[k] = acc
        fold_test_loss[k] = sum(test_loss) / len(test_loss)
        fold_train_loss[k] = train_Loss

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

    params['batch_size'] = str(params['batch_size'])  # stringify
    params['hidden_dim'] = str(params['hidden_dim'])  # stringify
    if idx_models == 0:  # create json files if model is the first
        model_ = {
            idx_models: {'params': params, 'acc': fold_acc, 'train_loss': fold_train_loss, 'test_loss': fold_test_loss}}
        dump_to_json(info_PATH, model_)
    else:  # dump to existing
        model_ = {
            idx_models: {'params': params, 'acc': fold_acc, 'train_loss': fold_train_loss, 'test_loss': fold_test_loss}}
        dump_to_existing_json(info_PATH, model_)


def CV_KFAC():
    pass



if __name__ == '__main__':
    pretrained_net_path = 'models/mnist/SWAG/model_200pretrained.pth'
    pretrained_opti_path = 'Optimizers/mnist/SWAG/opti_200pretrained.pth'

    which_data = 'mnist'
    K = 5

    params = {'device': "cuda" if torch.cuda.is_available() else "cpu",
              'batch_size': 128,
              'test_batch_size': 1000,
              'n_epochs': 100,
              'seed': 0,
              'hidden_dim': 200,
              'save_model': False,
              'l2': 0.001,
              'momentum': 0.001,
              'c': 1,
              'K': 20,
              'S': 20}

    if which_data == 'mnist':
        lr_range = torch.logspace(-2, 0, 30)

    optimise_swag(which_data=which_data, n_splits=K, lr_range=lr_range, params=params, net_path=pretrained_net_path,
                  opti_path=pretrained_opti_path, T_scale=False)
