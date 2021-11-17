import torch
import torch.nn as nn
from NN import Net
# detach to avoid affecting the original gradients.
# Clone keep requires grad equal to the cloned tensor
from utils import plot_decision_boundary, Squared_matrix
from torch.distributions.multivariate_normal import MultivariateNormal
from BMA import bma

def update_running_moment(theta, theta_i, n):
    """
    params: theta: tensor
            theta_i: tensor
            n: int
    """
    new = (n * theta + theta_i) / (n + 1)
    return new


class Swag_Net(Net):
    def __init__(self, input_dim, hidden_dim, output_dim, K, c, S, criterion, T, learning_rate, train_loader, test_loader, Xtest, ytest, l2_param):
        super().__init__(input_dim, hidden_dim, output_dim)

        self.K = int(K)
        self.p = sum(p.numel() for p in self.parameters())
        self.c = c
        self.S = S

        self.z1 = MultivariateNormal(torch.zeros(self.p), scale_tril=torch.diag(torch.ones(self.p)))
        self.z2 = MultivariateNormal(torch.zeros(self.K), scale_tril=torch.diag(torch.ones(self.K)))

        self.criterion = criterion
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        self.learning_rate = learning_rate
        self.l2_param = l2_param
        self.T = T

        self.D_hat = []
        self.theta_swa = torch.tensor([])
        self.sigma_vec = torch.tensor([])

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.ytest = ytest
        self.Xtest = Xtest

    def get_ith_moments(self):
        # Extract flattened parameters
        params = nn.utils.parameters_to_vector(self.parameters())

        # Initialize moments (second is uncentered)
        theta1 = params.clone().detach()  # detach from original tensor
        theta2 = params.clone().detach() ** 2

        return theta1, theta2

    def swag(self, net_path, opti_path):
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

        # Load pretrained network
        self.load_state_dict(torch.load(net_path))
        self.optimizer.load_state_dict(torch.load(opti_path))

        # Alter parameters
        for g in self.optimizer.param_groups:
            g['weight_decay'] = self.l2_param
            g['lr'] = self.learning_rate

        # plot pretrained network
        plot_decision_boundary(model = self, X =self.train_loader.dataset.X, y =self.train_loader.dataset.y, title="Pretrained")

        # Extract initial weights
        theta1, theta2 = self.get_ith_moments()

        # Initialize error and accuracy container
        test_loss, train_loss, all_acc = [], [], []
        n = 0

        # Run T epoch of training and testing
        for i in range(self.T):

            # Train swag
            n, theta1, theta2, loss_ = self.train_swag(epoch=i, n=n, theta1=theta1, theta2=theta2)

            # Collect train loss
            train_loss.append(loss_)

            # Test current SWAG model (if D hat has rank K)
            if len(self.D_hat) == self.K:
                self.theta_swa = theta1.clone().detach()
                self.sigma_vec = theta2 - theta1 ** 2
                _, acc, loss = bma(model = self, S = self.S, Xtest=self.Xtest, ytest=self.ytest, criterion=self.criterion)

                # collect test loss and accuracy
                test_loss.append(sum(loss) / len(loss))
                all_acc.append(sum(acc) / len(acc))

            else:
                test_loss.append(0)
                all_acc.append(0)

        # Update class values
        self.theta_swa = theta1.clone().detach()
        self.sigma_vec = theta2 - theta1 ** 2

        return train_loss, test_loss, all_acc

    def train_swag(self, epoch, n, theta1, theta2):
        loss_ave = 0
        for j, (Xtrain, ytrain, idx) in enumerate(self.train_loader):
            # Clear gradients w.r.t. parameters
            self.optimizer.zero_grad()

            # Forward pass to get scores
            outputs = self(Xtrain)

            # SGD update
            loss = self.criterion(outputs, ytrain)  # Cross entropy loss
            loss.backward()  # Gradients
            self.optimizer.step()  # Update parameters

            # Update loss average
            loss_ave = (loss_ave * j + loss.item()) / (j + 1)
            if j == 0:
                print("Running loss average: %s" %loss_ave)

        # Update moments with frequency c
        if epoch % self.c == 0 and epoch != 0:

            with torch.no_grad():

                # Update n
                n += 1

                # Get ith moment
                theta1_i, theta2_i = self.get_ith_moments()

                # Update first moment
                temp1 = update_running_moment(theta=theta1, theta_i=theta1_i, n=n)
                theta1 = temp1

                # Update second moment
                temp2 = update_running_moment(theta=theta2, theta_i=theta2_i, n=n)
                theta2 = temp2

                # Update D_hat
                if len(self.D_hat) == self.K:
                    self.D_hat.pop(0)  # remove first item
                self.D_hat.append(torch.unsqueeze(theta1_i - theta1, 1))

        #  Plot every 100th epoch
        if epoch % 10000 == 0 and len(self.D_hat) == self.K:
            plot_decision_boundary(model=self, X=self.Xtest, y=self.ytest, title ="Epoch: " + str(epoch) + " \n Moments computed " + str(n) + " times.",predict_func='bma')
        return n, theta1, theta2, loss_ave

    def sample_from_posterior(self):
        """
        input: m: SWA means in 1D
               Sigma_diag: sample vairances 1D
               D_hat: approximate sample covarinance 1D
        return: parameters in vector shape
        """
        Sigma_diag_squared = torch.diag(Squared_matrix(self.sigma_vec)) # TODO: Det går vel ikke rigtigt at tage abs???
        # Sample weight from posterior
        param_sample = self.theta_swa + 1 / (2 ** (1 / 2)) * torch.matmul(Sigma_diag_squared, self.z1.sample()) + 1 / (
                    (2 * (self.K - 1)) ** (1 / 2)) * torch.matmul(torch.cat(self.D_hat, dim=1), self.z2.sample())

        return param_sample
