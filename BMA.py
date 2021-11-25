import torch
import torch.nn as nn


def monte_carlo_bma(model, Xtest, ytest, S, C, forplot=False):

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
    :return: accuracy and loss for all models
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
            model.replace_network_weights(sampled_weights)

            # Monte Carlo
            p_yxw[i], score, _ = model.predict(Xtest)
            p_yx += 1 / S * p_yxw[i]


        if not forplot:
            yhat = torch.max(p_yxw[i], 1).indices
            acc = (yhat == ytest).sum() / len(ytest)
            accuracy.append(acc.item())
            loss = model.criterion(score, ytest)
            all_loss.append(loss.item())

            print('BMA loss.   Model number %i     loss: %s     accuracy: %s' %(i, loss, acc))

    if forplot:
        return p_yx

    else:
        return p_yxw, p_yx, accuracy, all_loss