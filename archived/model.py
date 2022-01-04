import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MyAwesomeModel(nn.Module):
    def __init__(self, train_loader, test_loader):
        super(MyAwesomeModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.n_epochs = None
        self.optimizer = None
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_losses = []
        self.train_counter = []
        self.test_losses = []
        self.test_counter = None

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

    # def _init(self):
    #     learning_rate = 0.01
    #     momentum = 0.5
    #     random_seed = 1
    #     torch.backends.cudnn.enabled = False
    #     torch.manual_seed(random_seed)
    #     self.optimizer = optim.SGD(self.parameters(), lr=learning_rate,
    #                                momentum=momentum)

    def fit(self, n_epochs: int = 3, lr: float = 0.01, momentum: float = 0.5, seed: int = None):
        torch.backends.cudnn.enabled = False
        if seed is not None:
            torch.manual_seed(seed)
        self.optimizer = optim.SGD(self.parameters(), lr=lr,
                                   momentum=momentum)
        self.n_epochs = n_epochs
        self.test_counter = [i * len(self.train_loader.dataset) for i in range(self.n_epochs + 1)]
        for epoch in range(1, self.n_epochs + 1):
            self._train(epoch)

    def _train(self, epoch):
        self.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            output = self(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            # if batch_idx % log_interval == 0:
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #     epoch, batch_idx * len(data), len(self.train_loader.dataset),
            #            100. * batch_idx / len(self.train_loader), loss.item()))
            self.train_losses.append(loss.item())
            self.train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(self.train_loader.dataset)))
            # torch.save(network.state_dict(), '/results/model.pth')
            # torch.save(optimizer.state_dict(), '/results/optimizer.pth')
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(self.train_loader.dataset),
                   100. * batch_idx / len(self.train_loader), loss.item()))

    def _test(self):

        self.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = network(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
