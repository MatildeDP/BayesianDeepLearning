import torch
import os
from copy import deepcopy
def monte_carlo_bma(model, Xtest, ytest, S, C, temp, criterion, forplot=False, save_models = '', path_to_bma = ''):

    """
    Monte carlo approximation of BMA.

    Notice, that the model class MUST contain following functions:
       1. sample_from_posterior
       2. replace_network_weights
       3. predict
       4. nn Criterion

    :param test_loader: test data
    :param S: number of models to sum over
    :param C: number of classes
    :param model: instance of KFAC or SWAG class
    :param Xtest: test data
    :param ytest: test labels
    :param temp: temperature scale parameter
    :param criterion: nn Cross entropy loss
    :param save_models: Path to models
    :param forplot: If True, method does not compute accuracy and only returns p_yx
    :param load_models: If not empty string, method loads models from path to compute BMA. If empty string, new models will be sampled
    :param param_idx: parameter count. Used to load correct models


    :return: p_yx: torch.tensor (nxc), contains p(y = c|x) for all classes and for all x in test_loader
    :return: p_yxw: dict. one key-value pair per model. Value = torch tensor (nxc), with (y=c|x,w), c in classes, x in test_loader
    :return: loss for each model
    :return: acc: accuracy of BMA prediction
    """

    # Deepcopy model, just in case
    model_ = deepcopy(model)

    n = len(Xtest)  # number of test points
    p_yx = torch.zeros(n, C)
    p_yxw = {i: [] for i in range(S)}
    accuracy, all_loss = [], []

    # compute bma with saved models
    if path_to_bma:
        # iterate through path
        for i, filename in enumerate(os.listdir(path_to_bma)):
            print(path_to_bma + filename)
            if '.DS' not in filename:
                model_.load_state_dict(torch.load(path_to_bma + filename))

                with torch.no_grad():
                    # Monte Carlo
                    p_yxw[i], score, _ = model_.predict(Xtest, temp=temp)
                    p_yx += 1 / S * p_yxw[i]

                    # TODO: implementing calibration for this one
                if not forplot:
                    loss = criterion(score, ytest)
                    all_loss.append(loss.item())
                    print('BMA loss.   Model number %i     loss: %s' %(i, loss))

    else:

        for i in range(S):

            with torch.no_grad():

                # Sample weights from posterior
                sampled_weights = model_.sample_from_posterior()

                # Replace network weights with sampled network weights
                test_model = model_.replace_network_weights(sampled_weights)

                # dump sampled_weights to json at path save_models
                if save_models:
                    torch.save(model_.state_dict(), save_models + 'bma_model_' + str(i))

                # Monte Carlo
                # if statement nessesary because SWAG return new model, KFAC does not
                if test_model is None:
                    p_yxw[i], score, _ = model_.predict(Xtest, temp = temp)

                else:
                    p_yxw[i], score, _ = test_model.predict(Xtest, temp = temp)

                p_yx += 1 / S * p_yxw[i]

                # TODO: implementing calibration for this one

            if not forplot:
                loss = criterion(score, ytest)
                all_loss.append(loss.item())
                #print('BMA loss.   Model number %i     loss: %s' %(i, loss))


    if forplot:
        return p_yx

    else:
        # compute overall accuracy
        yhat = torch.max(p_yx, 1).indices
        acc = (yhat == ytest).sum() / len(ytest)

        return p_yxw, p_yx, all_loss, acc
