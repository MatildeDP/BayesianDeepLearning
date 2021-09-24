import torch
import torch.nn as nn
from NN import DataLoaderInput, test, train, Net
from utils import plot_decision_boundary, plot_acc_and_loss
from sklearn.datasets import make_moons



if __name__ == '__main__':

    train_dataset = make_moons(n_samples=1000, noise=0.2, random_state=0)
    Xtrain, ytrain = train_dataset

    test_dataset = make_moons(n_samples=200, noise=0.2, random_state=0)
    Xtest, ytest = test_dataset

    batch_size = 128
    num_epochs = 200

    train_loader = torch.utils.data.DataLoader(dataset=DataLoaderInput(Xtrain, ytrain),
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=DataLoaderInput(Xtest, ytest),
                                              batch_size=batch_size,
                                              shuffle=False)

    input_dim = 2
    hidden_dim = 50
    output_dim = 2
    learning_rate = 0.1

    model = Net(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    model, train_loss, test_loss, acc = train(num_epochs = num_epochs,train_loader = train_loader, optimizer = optimizer, model = model, criterion = criterion, test_loader = test_loader)

    plot_decision_boundary(model, Xtest, ytest,title= "Decision boundary with test points on trained model")

    plot_acc_and_loss(testloss = test_loss, trainloss = train_loss, accuracy = acc)



