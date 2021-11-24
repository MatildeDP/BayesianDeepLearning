from utils import plot_decision_boundary
import torch
from NN import Net


class Deterministic_net(Net):
    def __init__(self, input_dim, hidden_dim, output_dim, lr_scheduler = None):
        super().__init__(input_dim, hidden_dim, output_dim)
        self.lr_scheduler = lr_scheduler

    def train_net(self, train_loader, optimizer, criterion, save_net = True, net_path = '', opti_path = ''):
        loss_ave = 0

        for i, (Xtrain, ytrain) in enumerate(train_loader):

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get scores
            outputs = self(Xtrain)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, ytrain)
            loss_ave = (loss_ave * i + loss.item()) / (i + 1)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Update parameters
            optimizer.step()


        # if epoch % 30 == 0:
        # plot decision boundary with train data
        # plot_decision_boundary(model, Xtrain,  ytrain, title = "Decision boundary with train points during training")

        if save_net:
            torch.save(self.state_dict(),  net_path)
            torch.save(optimizer.state_dict(), opti_path)


        return optimizer, criterion, loss_ave

    def test_net(self, test_loader, criterion, freq=0, epoch='', plot=False):
        self.eval()
        loss_ave, accuracy = 0, 0
        all_probs = []
        # Iterate through test dataset
        for j, (Xtest, ytest) in enumerate(test_loader):

            probs, score, pred = self.predict(Xtest)
            all_probs.append(probs)

            # Loss
            loss = criterion(score, ytest)
            loss_ave = (loss_ave * j + loss) / (j + 1)

            # Accuracy
            correct = (pred == ytest).sum() / len(ytest)
            accuracy = (accuracy * j + correct) / (j + 1)

            # plot decision boundary
            if plot:
                if type(epoch) == str:
                    plot_decision_boundary(self, Xtest, ytest, title="Decision boundary with test points")

                elif epoch % freq == 0 and j == 0:
                    plot_decision_boundary(self, Xtest, ytest,
                                           title="Decision boundary with test points: epoch %s" % epoch)

        all_probs = torch.cat(all_probs, dim=0)

        self.train()

        return accuracy, loss_ave, all_probs

    def run(self, num_epochs, test_loader, train_loader, criterion, optimizer, save_net = True, net_path = '', opti_path = ''):

        """
        Trains determnistic neural netowrk with structure defined by NN.Net
        :param num_epochs: Number of epochs to run
        :param test_loader: dataloader with test data
        :param train_loader: dataloader with train data
        :param criterion: nn loss
        :param optimizer: torch optim optimizer
        :param save_net: Boolean value
        :param net_path: Path to save net to
        :param opti_path: Path to save optimizer to
        :return: train_loss, test_loss and accuracy
        """
        test_loss = []
        train_loss = []
        acc = []
        for epoch in range(num_epochs):
            # Train
            optimizer, criterion, train_loss_ = self.train_net(train_loader, optimizer, criterion, save_net = save_net, net_path = net_path, opti_path = opti_path)

            # Test
            accuracy, tst_loss, _ = self.test_net(test_loader, criterion, freq=30, epoch=epoch)

            # Collect loss
            train_loss.append(float(train_loss_))
            test_loss.append(float(tst_loss.detach().numpy()))
            acc.append(accuracy)

            if epoch % 1 == 0:
                print(
                    'Iteration: {}. Test Loss: {}. Train Loss: {}. Accuracy: {}'.format(epoch, tst_loss.item(), train_loss_,
                                                                                        accuracy))

        return train_loss, test_loss, acc
