from deterministic import Deterministic_net
import torch.nn as nn
import torch
from sklearn.datasets import make_moons
from utils import plot_decision_boundary
from deterministic import Net
from torch.autograd import functional



class KFAC(Net):
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate, optimizer, L):
        super().__init__(input_dim, hidden_dim, output_dim)

        self.layers = ['fc1', 'fc2', 'fc3']
        self.L = L

        self.a = {i: [] for i in range(L)}  # activation dict running sum
        self.a_grad = {i: [] for i in range(1, L+1)}
        # TODO: Husk at der ikke er added ones til a endnu!!!
        self.h = {i: [] for i in range(1, L+1)}  # pre-activation dict running sum
        self.grads = {i: 0 for i in range(1, L+1)}  # gradients for each layer running sum
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
        self.D = self.MatrixContainer(L)
        self.B = self.MatrixContainer(L)


    def __setitem__(self, value, key = 0):
        self.a[key] += value

    class MatrixContainer:
        def __init__(self, L):
            self.data = {Lambda: 0 for Lambda in range(1, L+1)}
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
        #a3 = self.logsoftmax(h3)

        return h3

    def save_activation_grads(self, Lambda):
        def hook(mod, ind, out):
            """
            Hook structure
            :param mod: nn module
            :param ind: tensor
            :param out: tensor gradient
            """
            self.a_grad[Lambda].append((mod, out[0].detach()))
        return hook

    def save_pre_activation_grads(self,mod, ind, out):
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
            self.h[Lambda].append((mod, out[0].detach()))
        return hook


    def save_activations(self, Lambda):
        def hook(mod, ind, out):
            """
            Hook structure
            :param mod: nn module
            :param ind: tensor
            :param out: tensor gradient
            """

            self.a[Lambda].append((mod, out[0].detach()))

        return hook

    def load_model(self, net_path):
        self.load_state_dict(torch.load(net_path))

    def update_gradient_dict(self):
        self.grads[1] += torch.cat(
            (self.fc1.weight.grad, torch.reshape(self.fc1.bias.grad, (len(self.fc1.bias.grad), 1))), 1)

        self.grads[2] += torch.cat(
            (self.fc2.weight.grad, torch.reshape(self.fc2.bias.grad, (len(self.fc2.bias.grad), 1))), 1)

        self.grads[3] += torch.cat(
            (self.fc3.weight.grad, torch.reshape(self.fc3.bias.grad, (len(self.fc3.bias.grad), 1))), 1)

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

    def Hessian(self, func, input):
        """
        :param func: function to be evaluated
        :param input: tuple or tensor with input to func. Elements must be torch tensors
        :return: hessian of one data point
        """
        return functional.hessian(func, input)


    def collect_values(self, data):
        """
        Collects running sum of activations, pre-activations and gradients for each layer
        """
        for i, (Xtrain, ytrain, idx) in enumerate(data):
            # Clear gradients w.r.t. parameters
            self.optimizer.zero_grad()

            # store x = a0
            self.__setitem__([('a0', a0.detach()) for a0 in Xtrain])

            # Forward pass to get scores
            outputs = self(Xtrain)

            # Calculate Loss
            loss = self.criterion(outputs, ytrain)
            # loss_ave = (loss_ave * i + loss.item()) / (i + 1)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Collect weight gradients
            self.update_gradient_dict()


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
                    temp = (temp * n + torch.matmul(vec1, vec2))/(n+1)
                    n += 1

                self.Q[Lambda + 1] = [temp]
                Lambda += 1
            else:  # if data is not a N dimensional array
                pass
                # TODO: implement for higher dimensional data

    def compute_Qlambda(self, Lambda):
        """
        :param Lambda: Layer index
        :return: Covariance matrix of the activations
        """

        return torch.matmul(torch.unsqueeze(self.a[Lambda], 1), torch.unsqueeze(self.a[Lambda], 0))


    def compute_Hlambda(self, ddf_lambda, df_lambda, Lambda):

        # TODO: When you compute the hessian, remember that the functions only computes the hessian of one datapoint. you shall sum and divide with number of datapoint.

        if Lambda == self.L: # stop recursion after computation for 1st layer
            return



        pass

    def compute_Dlambda(self, ddf_lambda, Lambda):
        """
        :param ddf_lambda: second derivative of activation function
        :param Lambda: Layer index
        :return: Diagonal matrix with elements equal to the product of the second derivative of  the activation function
        w.r.t the pre-activation and the derivative of the cost function w.r.t the activation
        """
        return ddf_lambda(self.h[Lambda])*self.a_grad[Lambda] * torch.eye(len(self.h[Lambda]))


    def compute_B(self, df_lambda, Lambda):
        """
        :param df_lambda: derivative of activation function
        :param Lambda: Layer index
        :return: Diagonal matrix with elements equal to the derivative of the activation function w.r.t the pre-activation
        """
        return df_lambda(self.h[Lambda]) * torch.eye(len(self.h[Lambda]))


    def compute_Wstar(self):
        ##  TODO Mean of distirbution
        pass


from data import DataLoaderInput

if __name__ == '__main__':
    input_dim = 2
    hidden_dim = 4
    output_dim = 2
    learning_rate = 0.01

    net_path = 'models/NN_50.pth'
    opti_path = 'Optimizers/Opti_50.pth'

    model = Deterministic_net(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    #model.load_state_dict(torch.load(net_path)) # TODO Uncomment
    #optimizer.load_state_dict(torch.load(opti_path)) # TODO Uncomment

    train_dataset = make_moons(n_samples=100, noise=0.2, random_state=3)
    Xtrain, ytrain = train_dataset

    train_loader = torch.utils.data.DataLoader(dataset=DataLoaderInput(Xtrain, ytrain),
                                               batch_size=1,
                                               shuffle=True)

    # plot_decision_boundary(model=model, X=train_loader.dataset.X, y=train_loader.dataset.y, title="Pretrained")

    kfac_model = KFAC(input_dim, hidden_dim, output_dim, learning_rate, optimizer, 3)  # initialise class
    #kfac_model.load_model(net_path)  # load model  # TODO Uncomment
    kfac_model.collect_values(train_loader)
    kfac_model.compute_q()

    # Create normal distribution
    keys = ['fc1', 'fc2', 'fc3']
    items = [item for item in kfac_model.parameters()]
    idx = range(0, len(items), 2)

    # MAP solution for each layer (bias included)
    Wstar = {keys[i]: torch.cat((items[idx[i]], torch.reshape(items[idx[i] + 1], (len(items[idx[i]]), 1))), 1) for i in
             range(len(keys))}

    a = 123
