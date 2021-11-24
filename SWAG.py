import torch
import torch.nn as nn
from NN import Net
from utils import plot_decision_boundary, Squared_matrix
from torch.distributions.multivariate_normal import MultivariateNormal
from BMA import monte_carlo_bma

# TODO: Når parametre som skal tunes med "standard" grid search er tunet, skal skal de fixeres (conditioned on dastaset)
 # Hidden units, hidden layers, batch size, L2 parameter, momentum
def update_running_moment(theta, theta_i, n):
    """
    params: theta: tensor
            theta_i: tensor
            n: int
    """
    new = (n * theta + theta_i) / (n + 1)
    return new

class Swag(Net):
    def __init__(self, input_dim, hidden_dim, output_dim, K, c, S, criterion, num_epochs, learning_rate, l2_param):
        super().__init__(input_dim, hidden_dim, output_dim)

        self.K = int(K)
        self.p = sum(p.numel() for p in self.parameters())
        self.c = c
        self.S = S

        self.criterion = criterion
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        self.learning_rate = learning_rate
        self.l2_param = l2_param
        self.num_epochs = num_epochs

        self.D_hat = []
        self.theta_swa = torch.tensor([])
        self.sigma_vec = torch.tensor([])

        self.output_dim = output_dim


    def get_ith_moments(self):
        # Extract flattened parameters
        params = nn.utils.parameters_to_vector(self.parameters())

        # Initialize moments (second is uncentered)
        theta1 = params.clone().detach()  # detach from original tensor
        theta2 = params.clone().detach() ** 2

        return theta1, theta2

    def updateD(self, theta1_i, theta1):
        if len(self.D_hat) == self.K:
            self.D_hat.pop(0)  # remove first item
        self.D_hat.append(torch.unsqueeze(theta1_i - theta1, 1))

    def run_swag(self, net_path, opti_path, train_loader, test_loader, Xtest, ytest):
        """
        :param: net_path: Path to pre trained net
        :param: opti_path: Path to optimizer for pretrained net

        :return: m: SWA means in 1D
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
        #plot_decision_boundary(model = self, dataloader=train_loader,S = 20, title="Pretrained")

        # Extract initial weights
        theta1, theta2 = self.get_ith_moments()

        # Initialize error and accuracy container
        test_loss, train_loss, all_acc = [], [], []
        n = 0

        # Run T epoch of training and testing
        for i in range(self.num_epochs):

            # Train swag
            n, theta1, theta2, loss_ = self.train_swag(epoch=i, n=n, theta1=theta1, theta2=theta2, train_loader = train_loader, test_loader=test_loader)

            # Collect train loss
            train_loss.append(loss_)

            # Test current SWAG model (if D hat has rank K)
            if len(self.D_hat) == self.K:
                self.theta_swa = theta1.clone().detach()
                self.sigma_vec = theta2 - theta1 ** 2
                p_yxw, p_yx, acc, loss = monte_carlo_bma(model = self, S = self.S, Xtest=Xtest, ytest=ytest, C = self.output_dim)
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

    def train_swag(self, epoch, n, theta1, theta2, train_loader, test_loader):
        """
        :param epoch: Current epoch number
        :param n: number of updates to theta1, theta2 and Dhat
        :param theta1: current value of first moment
        :param theta2: current value of second moment

        :return: n
                 theta1: Updated first moment
                 theta2: Updated second moment
                 loss_ave: Train loss for epoch, averaged over batches
        """
        loss_ave = 0
        for j, (Xtrain, ytrain) in enumerate(train_loader):
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
                self.updateD(theta1_i, theta1)

        #  Plot every 100th epoch
        if epoch % 10000 == 0 and len(self.D_hat) == self.K:
            title = "Epoch: " + str(epoch) + " \n Moments computed " + str(n) + " times."
            plot_decision_boundary(model=self, dataloader = test_loader, title =title, predict_func='stochastic')
        return n, theta1, theta2, loss_ave

    def sample_from_posterior(self):
        """
        input: m: SWA means in 1D
               Sigma_diag: sample vairances 1D
               D_hat: approximate sample covarinance 1D
        return: parameters in vector shape
        """
        Sigma_diag_squared = torch.diag(Squared_matrix(abs(self.sigma_vec))) # TODO: Det går vel ikke rigtigt at tage abs???
        z1 = MultivariateNormal(torch.zeros(self.p), scale_tril=torch.diag(torch.ones(self.p)))
        z2 = MultivariateNormal(torch.zeros(self.K), scale_tril=torch.diag(torch.ones(self.K)))
        # Sample weight from posterior
        param_sample = self.theta_swa + 1 / (2 ** (1 / 2)) * torch.matmul(Sigma_diag_squared, z1.sample()) + 1 / (
                    (2 * (self.K - 1)) ** (1 / 2)) * torch.matmul(torch.cat(self.D_hat, dim=1), z2.sample())

        return param_sample

    def replace_network_weights(self, sampled_weights):
        nn.utils.vector_to_parameters(sampled_weights, self.parameters())







