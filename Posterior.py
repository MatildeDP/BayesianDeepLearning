import torch
import torch.nn as nn
from NN import Net
from data import DataLoaderInput
# detach to avoid affecting the original gradients.
# Clone keep requires grad equal to the cloned tensor
from utils import plot_decision_boundary, Squared_matrix, update_running_moment, plot_while_train
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal

class Swag_Net(Net):
    def __init__(self, input_dim, hidden_dim, output_dim, K):
        super().__init__(input_dim, hidden_dim, output_dim)

        self.K = int(K)
        self.p = sum(p.numel() for p in self.parameters())

        self.z1 = MultivariateNormal(torch.zeros(self.p), scale_tril=torch.diag(torch.ones(self.p)))
        self.z2 = MultivariateNormal(torch.zeros(self.K), scale_tril=torch.diag(torch.ones(self.K)))

    def swag(self, train_loader, T, optimizer, criterion, c, net_path, opti_path):
        """
        params: init_weight: Initial weights (for net). Pretrained on same data
                train_loader: Train data of type torch.utils.data.DataLoader
                test_loader: Test data of type torch.utils.data.DataLoader
                num_epochs: Num epochs to run when train() is called.

        return: m: SWA means in 1D
               Sigma_diag: sample vairances 1D
               D_hat: approximate sample covarinance 1D


        Stochastic weight average  - Gaussian
        """

        # Pretrain network
        # TODO: should i save and load optimizer as well?
        self.load_state_dict(torch.load(net_path))
        optimizer.load_state_dict(torch.load(opti_path))

        # plot pretrained network
        plot_while_train(train_loader=train_loader, net = self)

        # Extract initial weights
        theta1, theta2 = self.get_ith_moments()

        # Initialize D_hat
        D_hat = []
        train_loss = []

        # Train
        for i in range(T):
            loss_ave = 0
            for j, (Xtrain, ytrain, idx) in enumerate(train_loader):
                # Clear gradients w.r.t. parameters
                optimizer.zero_grad()

                # Forward pass to get scores
                outputs = self(Xtrain)

                # SGD update
                loss = criterion(outputs, ytrain)  # Cross entropy loss
                loss.backward() #  Gradients
                optimizer.step() # Update parameters

                # Update loss average
                loss_ave = (loss_ave * j + loss.item()) / (j + 1)

                # Update moments
            if i % c == 0 and i !=0:

                n = i / c
                print("n: %s, i: %s, c: %c" %(n,i,c))

                theta1_i, theta2_i = self.get_ith_moments()

                # Update first moment
                temp1 = update_running_moment(theta=theta1, theta_i=theta1_i, n=n)
                theta1 = temp1.clone().detach()

                # Update second moment
                temp2 = update_running_moment(theta=theta2, theta_i=theta2_i, n=n)
                theta2 = temp2.clone().detach()

                # Update D_hat
                if len(D_hat) == K:
                    D_hat.pop(0)  # remove first item
                D_hat.append(torch.unsqueeze(theta1_i - theta1, 1))

            train_loss.append(loss_ave)

        d_hat = torch.cat(D_hat, dim=1)
        theta_swa = theta1.clone().detach()
        sigma_vec = theta2 - theta1 ** 2

        # TODO: Den konvergerer mod en løsning uden varians.


        return theta_swa, sigma_vec, d_hat

    def get_ith_moments(self):
        # Extract flattened parameters
        params = nn.utils.parameters_to_vector(self.parameters())

        # Initialize moments (second is uncentered)
        theta1 = params.clone().detach()  # detach from original tensor
        theta2 = params.clone().detach() ** 2

        return theta1, theta2

    def sample_from_normal_posterior(self, m, Sigma_vec, D_hat):
        """
        input: m: SWA means in 1D
               Sigma_diag: sample vairances 1D
               D_hat: approximate sample covarinance 1D
        return: parameters in original shape
        """
        Sigma_diag_squared = torch.diag(Squared_matrix(Sigma_vec))
        # Sample weight from posterior
        param_sample = m + 1/(2**(1/2)) * torch.matmul(Sigma_diag_squared, self.z1.sample()) + 1/((2*(self.K-1))**(1/2)) * torch.matmul(D_hat, self.z2.sample())

        return param_sample

    def bma(self, S, thetaSWA, Sigma_vec, D_hat, X):
        """
        Monte Carlo approximation
        """
        self.eval()
        n = len(X)  # number of test points
        prob_dist = torch.zeros(n, 2)
        all_acc, all_loss = [], []
        for i in range(S):
            # Sample weights from posterior

            sampled_weights = self.sample_from_normal_posterior(m=thetaSWA, Sigma_vec=Sigma_vec, D_hat=D_hat)

            # Replace network weights with sampled weights
            nn.utils.vector_to_parameters(sampled_weights, self.parameters())

            # Monte Carlo
            outs = self(X)
            s = nn.Softmax(dim=1)
            probs = s(outs)
            prob_dist += 1 / S * probs



            # Store loss and accuracy
            #all_acc.append(accuracy)
            #all_loss.append(loss_ave)

            # TODO: Plot histogrammer af tilfældige predictive distributions
        self.train()

        return prob_dist


if __name__ == '__main__':
    train_dataset = make_moons(n_samples=1000, noise=0.3, random_state=3)
    Xtrain, ytrain = train_dataset

    test_dataset = make_moons(n_samples=200, noise=0.3, random_state=3)
    Xtest, ytest = test_dataset

    batch_size = 128
    num_epochs = 400

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
    c = 10
    K = num_epochs/c - 1



    model = Swag_Net(input_dim, hidden_dim, output_dim, K)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    thetaSWA, Sigma_vec, D_hat = model.swag(train_loader = train_loader, T = num_epochs, optimizer = optimizer, criterion = criterion, c=c, net_path= 'models/NN_10.pth', opti_path = 'Optimizers/opti_10.pth' )

    X = torch.tensor(Xtest, dtype = torch.float)
    y = torch.tensor(ytest, dtype = torch.float)
    pred_dist = model.bma(S = 10, thetaSWA = thetaSWA, Sigma_vec = Sigma_vec,D_hat = D_hat, X = X)
    plot_decision_boundary(model, X, y, thetaSWA = thetaSWA, Sigma_vec = Sigma_vec, D_hat = D_hat,title="SWAG", predict_func = 'bma')
    a = 123

#######################################
    # TODO: issues: er det data der gør at vi næsten for point estimates - eller er det fordi sgd ændrer på learning rate?


