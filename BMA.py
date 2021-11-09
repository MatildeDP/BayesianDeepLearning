import torch
import torch.nn as nn

def bma(model, S, Xtest, ytest, criterion, test = True):
    """
    Monte Carlo approximation
    """
    ytest = ytest
    n = len(Xtest)  # number of test points
    prob_dist = torch.zeros(n, 2)
    accuracy, all_loss = [], []

    for i in range(S):
        # Sample weights from posterior

        sampled_weights = model.sample_from_normal_posterior()

        # Replace network weights with sampled weights
        nn.utils.vector_to_parameters(sampled_weights, model.parameters())

        # Monte Carlo
        probs, preds, outs = model.predict(Xtest)
        prob_dist += 1 / S * probs


        if test:
            preds = torch.max(probs, 1).indices
            acc = (preds == ytest).sum() / len(ytest)
            accuracy.append(acc.item())
            loss = criterion(outs, ytest)
            all_loss.append(loss.item())


        # TODO: Plot histogrammer af tilfældige predictive distributions

    if test:
        a = sum(accuracy) / len(accuracy)
        print("                                            Average accuracy on test data %s " %a)

    return prob_dist, accuracy, all_loss