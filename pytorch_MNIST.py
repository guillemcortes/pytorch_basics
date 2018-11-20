import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) #in_channels=1 bc images are Black&White
        # 64 x 10 x image_size
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # 64 x 20 x image_size
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
            # 64 x 10 x 24 x 24 --> Nº images x Nº filters x
            # (Width - FilterSize + 2Padding)/Stride + 1 x
            # (Heigth - FilterSize + 2Padding)/Stride + 1
        x = F.relu(F.max_pool2d(self.conv2(x),2)) # 64 x 20 x 4 x 4
            # 64 x 10 x 8 x 8
        x = x.view(-1, 320) # Flatten | -1 = don't care about this dimension
        x = F.relu(self.fc1(x))
        return F.log_softmax(self.fc2(x), dim=1)


def train(device, optimizer):
    model.train()
    # model.train() tells your model that you are training the model. So
    # effectively layers like dropout, batchnorm etc. which behave different on
    # the train and test procedures know what is going on and hence can behave
    # accordingly.
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.


def main():
    args = parser.parse_args() # useful for passing arguments via command line
    device = torch.device("cpu")
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
