import utils
import data
import face_detection
from resnet import resnet
import torch
from torch import nn

DEBUG = False
BATCH_SIZE = 64
LR = 1e-3


def train_and_save_model():
    trainset, testset = data.get_train_test_set()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                              shuffle=False)
    net = resnet(in_channel=3, num_classes=1)
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    criterion = nn.BCELoss()
    utils.train(net, trainloader, testloader, 20, optimizer, criterion,
                debug=DEBUG,
                )


def test_model():
    face_detection.face_rec()


if __name__ == "__main__":
    test_model()


