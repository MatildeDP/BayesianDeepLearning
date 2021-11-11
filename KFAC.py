from deterministic import Deterministic_net
import torch.nn as nn
import torch
from sklearn.datasets import make_moons
from utils import plot_decision_boundary
from deterministic import Net
from torch.autograd import functional
from torch import matmul as matmul
from scipy.stats import matrix_normal
from scipy.linalg import det
from torch import inverse


class KFAC(Net):
    def __init__(self, input_dim, hidden_dim, output_dim, optimizer, L):
        super().__init__(input_dim, hidden_dim, output_dim)

        self.layers = {1: self.fc1, 2: self.fc2, 3: self.fc3}
        self.diff = {1: 'tanh', 2: 'tanh'}
        self.L = L

        self.a = {i: [] for i in range(L)}  # activation dict running sum
        self.a_grad = {i: [] for i in range(1, L + 1)}
        # TODO: Husk at der ikke er added ones til a endnu!!!
        self.h = {i: [] for i in range(1, L + 1)}  # pre-activation dict running sum
        self.grads = {i: 0 for i in range(1, L + 1)}  # gradients for each layer running sum
        self.dedh3 = 0

        self.optimizer = optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.tanh2 = nn.Tanh()

        # Backward hooks
        self.tanh.register_backward_hook(self.save_activation_grads(1))
        self.tanh2.register_backward_hook(self.save_activation_grads(2))
        self.fc3.register_backward_hook(self.save_pre_activation_grads)

        # Forward hooks
        self.fc1.register_forward_hook(self.save_pre_activations(1))
        self.fc2.register_forward_hook(self.save_pre_activations(2))
        self.fc3.register_forward_hook(self.save_pre_activations(3))
        self.tanh.register_forward_hook(self.save_activations(1))
        self.tanh2.register_forward_hook(self.save_activations(2))

        # Matrices
        self.Q = self.MatrixContainer(L)
        self.H = self.MatrixContainer(L)
        self.W = self.MatrixContainer(L)

    def __setitem__(self, value, key=0):
        self.a[key] = value

    class MatrixContainer:
        def __init__(self, L):
            self.data = {Lambda: 0 for Lambda in range(1, L + 1)}
            self.L = L

        def __getitem__(self, item):
            try:
                return self.data[item]
            except KeyError:
                print("The matrix index %s does not exist" % item)
                return None

        def __setitem__(self, key, value):
            self.data[key] = value



    def forward(self, x):
        x.requires_grad = True

        h1 = self.fc1(x)
        a1 = self.tanh(h1)
        h2 = self.fc2(a1)
        a2 = self.tanh2(h2)  # update activation
        h3 = self.fc3(a2)

        return h3

    def predict(self, X):
        score = self(X)
        s = nn.Softmax(dim=1)
        probs = s(score)

        return probs, score

    def tanh_ddf(self, x):
        return -2 * torch.tanh(x) * (1 - torch.tanh(x) ** 2)

    def tanh_df(self, x):
        return 1 - torch.tanh(x) ** 2

    def save_activation_grads(self, Lambda):
        def hook(mod, ind, out):
            """
            Hook structure
            :param mod: nn module
            :param ind: tensor
            :param out: tensor gradient
            """
            self.a_grad[Lambda] = (mod, out[0].detach())

        return hook

    def save_pre_activation_grads(self, mod, ind, out):
        """
        Hook structure
        :param mod: nn module
        :param ind: tensor
        :param out: tensor gradient
        """
        self.dedh3 = (mod, out[0].detach())

    def save_pre_activations(self, Lambda):
        def hook(mod, ind, out):
            """
            Hook structure
            :param mod: nn module
            :param ind: tensor
            :param out: tensor gradient
            """
            self.h[Lambda] = (mod, out[0].detach())

        return hook

    def save_activations(self, Lambda):
        def hook(mod, ind, out):
            """
            Hook structure
            :param mod: nn module
            :param ind: tensor
            :param out: tensor gradient
            """

            self.a[Lambda] = (mod, out[0].detach())

        return hook

    def load_model(self, net_path):
        self.load_state_dict(torch.load(net_path))

    def cross_entropy_loss_binary(self, correct, rest):
        """
        :param correct: The score of the network related to the correct class
        :param rest: The score of the network related to the incorrect class
        :return: cross entropy loss
        """

        loss = -torch.log(torch.exp(correct) / (torch.exp(rest) + torch.exp(correct)))
        return loss

    def cross_entropy_loss(self, scores):
        """
        :param scores: Output scores of network. First element MUST be the score of the correct class
        :return: cross entropy loss
        """
        return -torch.log(torch.exp(scores[0]) / (sum([torch.exp(i) for i in scores[1:]]) + torch.exp(scores[0])))

    def compute_hessian(self, func, input):
        """
        :param func: function to be evaluated
        :param input: tuple or tensor with input to func. Elements must be torch tensors
        :return: hessian of one data point
        """
        return functional.hessian(func, input)

    def collect_values(self, data):
        """
        Collects running average Q and H for each layer
        """
        # collect MAP estimate of weights
        self.collectWstar()

        for i, (Xtrain, ytrain, idx) in enumerate(data):
            # Clear gradients w.r.t. parameters
            self.optimizer.zero_grad()

            # store x = a0
            self.__setitem__([('a0', a0.detach()) for a0 in Xtrain][0])

            # Forward pass to get scores
            outputs = self(Xtrain)

            # Calculate Loss
            loss = self.criterion(outputs, ytrain)
            # loss_ave = (loss_ave * i + loss.item()) / (i + 1)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Compute Q and H
            self.compute_Q(n=i)
            self.compute_H(Lambda=1, n=i)

        a = 123

    def compute_q_old(self):
        """
        Computes the covariance of the activations

        """

        for Lambda, item in self.a.items():
            length = item[0][1]
            temp = torch.zeros(len(length), len(length))
            n = 0

            if len(length.shape) == 1:  # if data is an 1D array
                for _, act in item:
                    vec1 = torch.unsqueeze(act, 1)
                    vec2 = torch.unsqueeze(act, 0)
                    temp = (temp * n + torch.matmul(vec1, vec2)) / (n + 1)
                    n += 1

                self.Q[Lambda + 1] = [temp]
                Lambda += 1
            else:  # if data is not a N dimensional array
                pass
                # TODO: implement for higher dimensional data

    def compute_Q(self, n):
        """
        :param Lambda: Layer index
        :param n: number of datapoint iterated so far
        :return: Covariance matrix of the activations
        """
        for Lambda in self.Q.data.keys():
            self.Q[Lambda] = (self.Q[Lambda] * n + torch.matmul(
                torch.cat((torch.unsqueeze(self.a[Lambda - 1][1], 1), torch.tensor([[1]])), 0),
                torch.cat((torch.unsqueeze(self.a[Lambda - 1][1], 0),
                           torch.tensor([[1]])), 1))) / (n + 1)

    def compute_D(self, func, Lambda):

        """
        :param ddf_lambda: second derivative of activation function
        :param Lambda: Layer index
        :return: Diagonal matrix with elements equal to the product of the second derivative of  the activation function
        w.r.t the pre-activation and the derivative of the cost function w.r.t the activation
        """

        if func == 'tanh':
            return self.tanh_ddf(self.h[Lambda][1]) * self.a_grad[Lambda][1] * torch.eye(len(self.h[Lambda][1]))

    def compute_B(self, func, Lambda):
        """
        :param df_lambda: derivative of activation function
        :param Lambda: Layer index
        :return: Diagonal matrix with elements equal to the derivative of the activation function w.r.t the pre-activation
        """
        if func == 'tanh':
            return self.tanh_df(self.h[Lambda][1]) * torch.eye(len(self.h[Lambda][1]))

            # return torch.cat((self.tanh_df(self.h[Lambda][1]), torch.zeros(1)), 0) * torch.eye(
            #    len(self.h[Lambda][1]) + 1)

    def compute_H(self, Lambda, n):
        """
        Computes (for one point) and stores (running average over points) the pre-activation hessian for layer Lambda
        :param Lambda: Layer number
        :return:
        """
        if Lambda == self.L:  # base case
            self.H[Lambda] = (self.H[Lambda] * n + self.compute_hessian(self.cross_entropy_loss, self.h[Lambda][1])) / (
                    n + 1)
            return self.H[Lambda]

        diff_ = self.diff[Lambda]
        B = self.compute_B(diff_, Lambda)
        W = self.layers[Lambda + 1].weight.detach()  # ,torch.unsqueeze(self.layers[Lambda + 1].bias.detach(), 1)), 1)
        Wt = torch.transpose(W, 0, 1)
        D = self.compute_D(diff_, Lambda)
        # Lambda += 1

        self.H[Lambda] = (self.H[Lambda] * n + (matmul(B, matmul(Wt, matmul(
            self.compute_H(Lambda=Lambda + 1, n=1), matmul(W, B)))) + D)) / (n + 1)

        return self.H[Lambda]

    def collectWstar(self):
        for Lambda in self.W.data.keys():
            l = self.layers[Lambda].weight.detach()
            b = self.layers[Lambda].bias.detach()
            self.W[Lambda] = torch.cat((l, torch.unsqueeze(b, 1)), 1)

    def regularize_and_add_prior(self, tau, N):
        """
        :param tau: Precision of Gaussian prior
        :param N: size of dataset
        :return:
        """

        for Lambda in self.H.data.keys():
            sqrt_tau = tau ** (1 / 2)
            sqrt_N = N ** (1 / 2)

            IH = len(self.H[Lambda])
            self.H[Lambda] = self.H[Lambda] * sqrt_N + sqrt_tau * torch.eye(IH)

            IQ = len(self.Q[Lambda])
            self.Q[Lambda] = self.Q[Lambda] * sqrt_N + sqrt_tau * torch.eye(IQ)

    def sample_from_posterior(self):
        """
        :return: weight samples
        """

        # Initialise sample dict
        samples = {Lambda: None for Lambda in range(1, self.L + 1)}

        for Lambda in samples.keys():
            # Invert covariances
            Q_inv = inverse(self.Q[Lambda])
            H_inv = inverse(self.H[Lambda])

            # Sample
            samples[Lambda] = matrix_normal.rvs(mean=self.W[Lambda], rowcov=H_inv, colcov=Q_inv, size=1)

        return samples

    def monte_carlo_bma(self, test_loader, S, C):
        """
        :param test_loader: test data
        :param S: number of models to sum over
        :param m: number of features
        :return: p_yx: torch.tensor (nxc), contains p(y = c|x) for all classes and for all x in test_loader
        :return: p_yxw: dict. one key-value pair per model. Value = torch tensor (nxc), with (y=c|x,w), c in classes, x in test_loader
        :return: accuracy and loss for all models
        """

        n = len(test_loader)  # number of test points
        p_yx = torch.zeros(n, C)
        p_yxw = {i: [] for i in range(S)}
        accuracy, all_loss = [], []
        Xtest, ytest = torch.tensor(test_loader.dataset.X), torch.tensor(test_loader.dataset.y)

        for i in range(S):

            # Sample weights from posterior
            sampled_weights = self.sample_from_posterior()

            # Replace network weights with sampled weights
            for idx in range(1, len(self.layers) + 1):
                w_idx = self.layers[idx].weight.shape

                self.layers[idx].weight.data = torch.tensor(sampled_weights[idx][:, :w_idx[1]])
                self.layers[idx].bias.data = torch.squeeze(torch.tensor(sampled_weights[idx][:, w_idx[1]:]))

            # Monte Carlo
            p_yxw[i], score = self.predict(Xtest)
            p_yx += 1 / S * p_yxw[i]

            preds = torch.max(p_yxw[i], 1).indices
            acc = (preds == ytest).sum() / len(ytest)
            accuracy.append(acc.item())
            loss = criterion(score, ytest)
            all_loss.append(loss.item())

        return p_yxw, p_yx, accuracy, all_loss


from data import DataLoaderInput

if __name__ == '__main__':
    input_dim = 2
    hidden_dim = 50
    output_dim = 2
    learning_rate = 0.1
    tau = 0.1
    noise = 0.8

    net_path = 'models/NN_50.pth'
    opti_path = 'Optimizers/Opti_50.pth'

    model = Deterministic_net(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    model.load_state_dict(torch.load(net_path)) # TODO load model
    optimizer.load_state_dict(torch.load(opti_path)) # TODO load model

    train_dataset = make_moons(n_samples=100, noise=noise, random_state=3)
    Xtrain, ytrain = train_dataset

    train_loader = torch.utils.data.DataLoader(dataset=DataLoaderInput(Xtrain, ytrain),
                                               batch_size=1,
                                               shuffle=True)

    test_dataset = make_moons(n_samples=20, noise=noise, random_state=5)
    Xtest, ytest = test_dataset

    test_loader = torch.utils.data.DataLoader(dataset=DataLoaderInput(Xtrain, ytrain),
                                              batch_size=1,
                                              shuffle=False)

    # plot_decision_boundary(model=model, X=train_loader.dataset.X, y=train_loader.dataset.y, title="Pretrained")

    kfac_model = KFAC(input_dim, hidden_dim, output_dim, optimizer, 3)  # initialise class
    kfac_model.load_model(net_path)  # TODO load model
    kfac_model.collect_values(train_loader)
    kfac_model.regularize_and_add_prior(tau=tau, N=len(train_loader))
    p_yxw, p_yx, accuracy, all_loss = kfac_model.monte_carlo_bma(test_loader, S=10, C=2)

    a = 123
