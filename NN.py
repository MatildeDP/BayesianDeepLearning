import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import plot_decision_boundary
from copy import copy

# TODO implement running average instead of storing lists for loss!!!

class DataLoaderInput:
    """
    Data structure required for torch data iterable
    """
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.tensor(self.y[idx])
        return x, y, idx

    def __len__(self):
        return len(self.X)

class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        return x

    def predict(self, X):
        outs = self(X)
        s = nn.Softmax(dim=1)
        probs = s(outs)

        return probs, torch.max(probs, 1).indices, outs



def test(test_loader, model, criterion, freq = 0, epoch = ''):
    correct = 0
    total = 0
    test_loss = []
    # Iterate through test dataset
    for j, (Xtest, ytest, idx_test) in enumerate(test_loader):

        probs, pred, score = model.predict(Xtest)

        loss = criterion(score, ytest)
        test_loss.append(loss)

        # Total number of labels
        total += ytest.size(0)

        # Total correct predictions
        correct += (pred == ytest).sum()

        # plot decision boundary
        #if type(epoch) == str:
         #   plot_decision_boundary(model, Xtest, ytest, title="Decision boundary with test points")

        #elif epoch % freq == 0 and j == 0:
        #    plot_decision_boundary(model, Xtest, ytest, title="Decision boundary with test points: epoch %s" % epoch)

    accuracy = 100 * correct / total
    return accuracy, sum(test_loss)/len(test_loss)

def train(num_epochs,train_loader, test_loader, optimizer, model, criterion):
    test_loss = []
    train_loss = []
    acc = []
    for epoch in range(num_epochs):
        loss_ = []
        for i, (Xtrain, ytrain, idx) in enumerate(train_loader):

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs = model(Xtrain)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, ytrain)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()

            loss_.append(loss.item())
            # compute running average instead

        #if epoch % 30 == 0:
            # plot decision boundary with train data
            #plot_decision_boundary(model, Xtrain,  ytrain, title = "Decision boundary with train points during training")


        # Test on test data
        accuracy, tst_loss = test(test_loader, model, criterion, freq = 30, epoch = epoch)

        # Print Loss
        train_loss_mean = sum(loss_) / len(loss_)
        print('Iteration: {}. Test Loss: {}. Train Loss: {}. Accuracy: {}'.format(epoch, tst_loss.item(), train_loss_mean, accuracy))

        train_loss.append(float(train_loss_mean))
        test_loss.append(float(tst_loss.detach().numpy()))
        acc.append(accuracy)

    return model, train_loss, test_loss, acc










