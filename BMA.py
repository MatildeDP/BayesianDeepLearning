import torch
import torch.nn as nn


def monte_carlo_bma(model, Xtest, ytest, S, C, forplot=False, save_models = ''):

    """
    Monte carlo approximation of BMA.

    Notice, that the model class MUST contain following functions:
       1. sample_from_posterior
       2. replace_network_weights
       3. predict
       4. nn Criterion

    :param test_loader: test data
    :param S: number of models to sum over
    :param c: number of classes

    :return: p_yx: torch.tensor (nxc), contains p(y = c|x) for all classes and for all x in test_loader
    :return: p_yxw: dict. one key-value pair per model. Value = torch tensor (nxc), with (y=c|x,w), c in classes, x in test_loader
    :return: loss for each model
    :return: acc: accuracy of BMA prediction
    """

    n = len(Xtest)  # number of test points
    p_yx = torch.zeros(n, C)
    p_yxw = {i: [] for i in range(S)}
    accuracy, all_loss = [], []

    for i in range(S):

        with torch.no_grad():

            # Sample weights from posterior
            sampled_weights = model.sample_from_posterior()

            # Replace network weights with sampled network weights
            test_model = model.replace_network_weights(sampled_weights)

            # dum sampled_weights to json at path save_models
            #if save_models:


            # Monte Carlo
            p_yxw[i], score, _ = test_model.predict(Xtest)
            p_yx += 1 / S * p_yxw[i]



        if not forplot:
            loss = model.criterion(score, ytest)
            all_loss.append(loss.item())
            #print('BMA loss.   Model number %i     loss: %s' %(i, loss))


    if forplot:
        return p_yx

    else:
        # compute overall accuracy
        yhat = torch.max(p_yx, 1).indices
        acc = (yhat == ytest).sum() / len(ytest)
        #print("BMA predictions accuracy %s" %acc)
        return p_yxw, p_yx, all_loss, acc
