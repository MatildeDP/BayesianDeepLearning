import torch
import torch.nn as nn
from deterministic import Deterministic_net
from data import DataLoaderInput
from utils import plot_decision_boundary, plot_acc_and_loss
from sklearn.datasets import make_moons, make_circles



def main_deterministic():
    train_dataset = make_moons(n_samples=1000, noise=0.1, random_state=3)
    Xtrain, ytrain = train_dataset

    test_dataset = make_moons(n_samples=200, noise=0.1, random_state=3)
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
    hidden_dim = 10
    output_dim = 2
    learning_rate = 0.1

    model = Deterministic_net(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    ############
    model.load_state_dict(torch.load('models/NN_10_circle.pth'))
    optimizer.load_state_dict(torch.load('Optimizers/opti_10_circle.pth'))
    #############

    train_loss, test_loss, acc = model.run(num_epochs, test_loader, train_loader,criterion, optimizer, save_net = False, net_path = 'NN_10_circle.pth', opti_path = 'opti_10_circle.pth')

    plot_decision_boundary(model, Xtest, ytest, title="Decision boundary with test points on trained model")

    plot_acc_and_loss(testloss=test_loss, trainloss=train_loss, accuracy=acc)


if __name__ == '__main__':

    # Run deterministic net
    main_deterministic()


