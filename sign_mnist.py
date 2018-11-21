# Import modules
import argparse
import random
from copy import deepcopy
from pathlib import Path
from math import floor
import numpy as np
import pandas as pd
# import matplotlib
# matplotlib.use('PS')
# from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import Dataset

from torch import nn
from torch.nn import functional as F
from torch import optim

from torchvision import models
from torchvision import transforms


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5) #in_channels=1. B&W images
        # batch_size(32) x 10 x image_size
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # batch_size(32) x 20 x image_size
        # [32, 20*4*4] = [32, 320]
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 26) #26 classes
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must
        return a Variable of output data. We can use Modules defined in the
        constructor as well as arbitrary operators on Variables.
        """
        # 28 x 28 images
        x = F.relu(F.max_pool2d(self.conv1(x),2)) # batch_size x 10 x 12 x 12
            # Nº images x Nº filters x Width x Height
            # cnn output size = (input_size - kernel_size + 2*padding)/stride + 1
            # batch_size x 10 x (28-5+2*0)/1+1 x (28-5+2*0)/1+1 = 32 x 10 x 24 x 24
            # after pooling: 24 / 2 = 12
            # output dim = 32 x 10 x 12 x 12 
        x = F.relu(F.max_pool2d(self.conv2(x),2)) # 32 x 20 x 4 x 4
            # 32 x 20 x 8 x 8
        x = x.view(-1, 320) # Flatten | -1 = don't care about this dimension
        x = F.relu(self.fc1(x))
        return self.softmax(self.fc2(x))
        #return F.log_softmax(self.fc2(x), dim=1)
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
        # TODO: investigate on this.
        loss = F.nll_loss(output, target) # -log() + softmax
        #loss = nn.NLLLoss(output, target) # -log() + softmax
        loss.backward()  # compute gradients
        optimizer.step() # proceed gradient descent
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def validation(device, model, validation_loader):
    # test is in reality VALIDATION SET!
    model.eval()
    validation_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            validation_loss += F.nll_loss(output, target, reduction='sum').item()
            # sum of the test_loss in order to compute the AVG later
            # TODO: draw the output vector of each sentence
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    validation_loss /= len(validation_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        validation_loss, correct, len(validation_loader.dataset),
        100. * correct / len(validation_loader.dataset)))

def display_data(dfs, labels, images, tensors):
    train_df, test_df = dfs
    train_labels, test_labels = labels
    train_images, test_images = images
    train_images_tensor, train_labels_tensor, test_images_tensor, test_labels_tensor = tensors

    print('train_df dataframe summary: {}'.format(train_df.head(2)))
    print('test_df dataframe summary: {}'.format(test_df.head(2)))

    print('train_labels type is {}'.format(type(train_labels)))
    print('test_labels type is {}'.format(type(test_labels)))

    print('train_images type is {} with shape {}'.format(type(train_images), train_images.shape))
    print('test_images type is {} with shape {}'.format(type(test_images), test_images.shape))

    print('train_images_tensor type is {} with shape {}'.format(type(train_images_tensor), train_images_tensor.shape))
    print('train_labels_tensor type is {} with shape {}'.format(type(train_labels_tensor), train_labels_tensor.shape))
    print('test_images_tensor type is {} with shape {}'.format(type(test_images_tensor), test_images_tensor.shape))
    print('test_labels_tensor type is {} with shape {}'.format(type(test_labels_tensor), test_labels_tensor.shape))


def main():
    desc = "Pytorch implementation of a Sign-MNIST challenge solution"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=5,
                        help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='The Learning Rate')
    parser.add_argument('--device', type=str, default="cpu",
                        help='cpu/gpu')
    args = parser.parse_args()

    # Dataset path
    dataset_path = Path('../datasets/sign-language-mnist')
    train_csv_path = dataset_path.joinpath('sign_mnist_train.csv')
    test_csv_path = dataset_path.joinpath('sign_mnist_test.csv')

    # Load CSV data as dataframes
    train_df = pd.read_csv(str(train_csv_path))
    test_df = pd.read_csv(str(test_csv_path))
    # csv = label, pixel1, pixel2, ..., pixel784 (28x28 images)

    # Get labels
    train_labels = train_df['label'].values
    test_labels = test_df['label'].values

    # Drop labels from dataframes
    train_df.drop('label', axis = 1, inplace = True)
    test_df.drop('label', axis = 1, inplace = True)

    # Resahpe images to 1x28x28 (Channels x Width x Height) into a numpy array
    train_images = np.array([i.reshape(1, 28, 28) for i in (train_df.values.astype(np.uint8))])
    test_images = np.array([i.reshape(1, 28, 28) for i in (test_df.values.astype(np.uint8))])

    # Convert numpy to Tensor
    train_images_tensor = torch.FloatTensor(train_images)
    train_labels_tensor = torch.LongTensor(train_labels)
    test_images_tensor = torch.FloatTensor(test_images)
    test_labels_tensor = torch.LongTensor(test_labels)


    dfs = train_df, test_df
    labels = train_labels, test_labels 
    images = train_images, test_images  
    tensors = train_images_tensor, train_labels_tensor, test_images_tensor, test_labels_tensor 

    if args.verbose:
        display_data(dfs, labels, images, tensors)

    train_set = TensorDataset(train_images_tensor, train_labels_tensor)
    test_set = TensorDataset(test_images_tensor, test_labels_tensor)

    # Split train set into train and validation subsets
    num_examples = len(train_set)
    num_validation_examples = floor(0.3*len(train_set))
    num_train_examples = len(train_set) - num_validation_examples
    print(f'Number of examples: {num_examples} \n'
          f'Number of train examples: {num_train_examples} \n'
          f'Number of validation examples: {num_validation_examples}')

    subsets = random_split(train_set, [num_train_examples, num_validation_examples])
    train_subset = subsets[0]
    validation_subset = subsets[1]

    # Dataloaders
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_subset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    model = SimpleCNN()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    device = torch.device(args.device)

    for epoch in range(1, args.epochs+1):
        train(device, model, optimizer, train_loader, epoch, args.log_interval)
        validation(device, model, validation_loader)

if __name__ == '__main__':
    main()
