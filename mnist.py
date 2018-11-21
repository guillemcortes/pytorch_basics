import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) #in_channels=1. B&W images
        # 64(batch_size) x 10 x image_size
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # 64(batch_size) x 20 x image_size
        # [64, 20*4*4] = [64, 320]
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must
        return a Variable of output data. We can use Modules defined in the
        constructor as well as arbitrary operators on Variables.
        """
        # 28 x 28
        x = F.relu(F.max_pool2d(self.conv1(x),2)) # 64 x 10 x 12 x 12
            # 64 x 10 x (24 x 24)/pooling --> Nº images x Nº filters x
            # (Width - FilterSize + 2Padding)/Stride + 1 x
            # (Heigth - FilterSize + 2Padding)/Stride + 1
        x = F.relu(F.max_pool2d(self.conv2(x),2)) # 64 x 20 x 4 x 4
            # 64 x 10 x (8 x 8)/pooling 
        x = x.view(-1, 320) # Flatten | -1 = don't care about this dimension
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)
        # if we define de softmax here, then we should use the nll_loss instead
        # of the cross_entropy


def train(device, model, optimizer, train_loader, epoch, log_interval):
    model.train()
    # model.train() tells your model that you are training the model. So
    # effectively layers like dropout, batchnorm etc. which behave different on
    # the train and test procedures know what is going on and hence can
    # behave accordingly.
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # clears the gradients of all optimized tensors.
                              # thats why loss.backward() and optimizer.step()
                              # are separated.
        output = model(data)
        loss = F.nll_loss(output, target) # -log() + softmax
        loss.backward()  # compute gradients
        optimizer.step() # proceed gradient descent
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(device, model, test_loader):
    # test is in reality VALIDATION SET!
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # sum of the test_loss in order to compute the AVG later
            # TODO: draw the output vector of each sentence
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    # HYPERPARAMETERS --> how to choose them?
    seed = 1
    train_batch_size = 64
    test_batch_size = 1000
    epochs = 10
    lr = 0.01
    momentum = 0.5
    log_interval = 10 #log every 10 batches
    device = torch.device("cpu")

    # Hyperparameters can be passed as cmd arguments as follow
    # parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # parser.add_argument('--batch-size', type=int, default=64, metavar='Nº',
    #                     help='input batch size for training (default: 64)')
    # args = parser.parse_args() # capture arguments via command line

    torch.manual_seed(seed) #for reproducibility

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            root='../MNIST_data',
            train=True,
            download=False,
            transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
            ])), # normalize with precomputed mean and std
        batch_size=train_batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='../MNIST_data',
                       train=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])), # normalize with precomputed mean and std
        batch_size=test_batch_size,
        shuffle=True,
    )

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    for epoch in range(1, epochs+1):
        train(device, model, optimizer, train_loader, epoch, log_interval)
        test(device, model, test_loader)


if __name__ == '__main__':
    main()
