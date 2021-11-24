import torch
from sklearn.datasets import make_moons
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)

class DataLoaderInput:
    """
    Data structure required for torch data iterable
    """

    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        x = self.X[idx].float()
        y = self.y[idx]
        return x, y

    def __len__(self):
        return len(self.X)


class LoadDataSet:
    def __init__(self, dataset):
        self.dataset = dataset
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([1,784])])

    def load_training_data(self, batch_size, noise=None, n_samples=1000):

        if self.dataset == 'two_moons':
            train_dataset = make_moons(n_samples=n_samples, noise=noise, random_state=3)
            Xtrain, ytrain = train_dataset
            Xtrain, ytrain = torch.tensor(Xtrain, dtype=torch.float32), torch.tensor(ytrain)

            train_data = DataLoaderInput(Xtrain, ytrain)

        elif self.dataset == "mnist":
            train_data = datasets.MNIST(
                root='data',
                train=True,
                transform=transforms.ToTensor(),
                download=True)

            Xtrain, ytrain = train_data.data.reshape(-1, 28*28), train_data.targets
            train_data = DataLoaderInput(Xtrain, ytrain)

        # Create dataloader object
        train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size,
                                                   shuffle=True)


        return Xtrain, ytrain, train_loader

    def load_test_data(self, batch_size, noise=None, n_samples=200):

        if self.dataset == 'two_moons':
            test_dataset = make_moons(n_samples=n_samples, noise=noise, random_state=3)
            Xtest, ytest = test_dataset
            Xtest, ytest = torch.tensor(Xtest, dtype=torch.float32), torch.tensor(ytest)
            test_data = DataLoaderInput(Xtest, ytest)


        elif self.dataset == "mnist":
            test_data = datasets.MNIST(
                root='data',
                train=False,
                transform=transforms.ToTensor()
            )

            Xtest, ytest = test_data.data.reshape(-1, 28*28), test_data.targets
            test_data = DataLoaderInput(Xtest, ytest)


        # Create dataloader object
        test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                                  batch_size=batch_size,
                                                  shuffle=False)

        return Xtest, ytest, test_loader
