from utils import train
import data
from resnet import resnet
import torch
from torch import nn

DEBUG = False
BATCH_SIZE = 64
LR = 1e-3

if __name__ == "__main__":
    trainset, testset = data.get_train_test_set()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                              shuffle=False)
    net = resnet(in_channel=3, num_classes=1)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    criterion = nn.BCELoss()
    train(net, trainloader, testloader, 20, optimizer, criterion,
          debug=DEBUG,
          )
