from datetime import datetime
import os
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

SAVE_ROOT = "../data/experiment"


def get_acc(output, label):
    total = output.shape[0]
    pred_label = (output > 0.5)
    num_correct = (pred_label.int() == label.int()).sum().item()
    return num_correct / total


def check_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def train(net, train_data, valid_data, num_epochs,
          optimizer, criterion, saveRoot=None, dirName=None, debug=False):
    trainLossHistory = []
    trainAccHistory = []
    validLossHistory = []
    validAccHistory = []
    bestValidAcc = 0
    useCuda = torch.cuda.is_available()
    if useCuda:
        net = net.cuda()
    prev_time = datetime.now()
    if saveRoot is None:
        saveRoot = SAVE_ROOT
    if dirName is None:
        dirName = str(prev_time)[:-10].replace(":", ".")
    saveDir = os.path.join(saveRoot, dirName)
    check_dir(saveDir)

    for epoch in range(num_epochs):
        train_loss = 0
        train_acc = 0
        net = net.train()
        data_stream = tqdm(enumerate(train_data))
        # 训练一个epoch
        for batch_index, (im, label) in data_stream:
            if useCuda:
                im = Variable(im.cuda())  # (bs, 3, h, w)
                label = Variable(label.cuda())  # (bs, h, w)
            else:
                im = Variable(im)
                label = Variable(label)
            # forward
            output = net(im)
            loss = criterion(output, label.float())
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            current_acc = get_acc(output, label)
            train_acc += current_acc
            
            trainLossHistory.append(loss.item())
            trainAccHistory.append(current_acc)
            
            data_stream.set_description((
                    "=>"
                    "epoch: {epoch}/{num_epochs} | "
                    "progress: {batch_index}/{batch_indexs} | "
                    "train_loss: {train_loss} | "
                    "train_acc: {train_acc} | "
                    ).format(
                    epoch=epoch,
                    num_epochs=num_epochs,
                    batch_index=batch_index,
                    batch_indexs=len(train_data),
                    train_loss=loss.item(),
                    train_acc=current_acc,
                            ))
            if debug and batch_index > 5:
                break

        # 开始验证
        valid_loss = 0
        valid_acc = 0
        net = net.eval()
        for batch_index, (im, label) in enumerate(valid_data):
            if useCuda:
                im = Variable(im.cuda())
                label = Variable(label.cuda())
            else:
                im = Variable(im)
                label = Variable(label)
            output = net(im)
            loss = criterion(output, label.float())
            valid_loss += loss.item()
            valid_acc += get_acc(output, label)
            if debug and batch_index > 5:
                break

        validLossHistory.append(valid_loss / len(valid_data))
        validAccHistory.append(valid_acc / len(valid_data))
        epoch_str = (
            "Epoch %d. Train Loss: %f, Train Acc: %f, Valid Loss: %f, Valid Acc: %f, "
            % (epoch, train_loss / len(train_data),
               train_acc / len(train_data), valid_loss / len(valid_data),
               valid_acc / len(valid_data)))

        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        prev_time = cur_time
        if not debug:
            print(epoch_str + time_str)

        torch.save(trainLossHistory, os.path.join(saveDir, "trainLossHistory"))
        torch.save(validLossHistory, os.path.join(saveDir, "validLossHistory"))
        torch.save(trainAccHistory, os.path.join(saveDir, "trainAccHistory"))
        torch.save(validAccHistory, os.path.join(saveDir, "validAccHistory"))
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            "useCuda": useCuda
            },
            os.path.join(saveDir, "latest_ckpt"),
        )
        if validAccHistory[-1] > bestValidAcc:
            bestValidAcc = validAccHistory[-1]
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                "useCuda": useCuda
                },
                os.path.join(saveDir, "best_ckpt"),
            )
    plot_loss_acc(saveDir)


def plot_loss_acc(saveDir, title="loss_accuracy"):
    trainLossHistory = torch.load(os.path.join(saveDir, "trainLossHistory"))
    trainAccHistory = torch.load(os.path.join(saveDir, "trainAccHistory"))
    validLossHistory = torch.load(os.path.join(saveDir, "validLossHistory"))
    validAccHistory = torch.load(os.path.join(saveDir, "validAccHistory"))

    epochNum = len(validLossHistory)
    stepPerEpoch = len(trainLossHistory)//epochNum

    fig, axs = plt.subplots(2, 1, figsize=(18, 10))
    axs[0].set_xlabel('epoch')
    axs[0].set_ylabel('loss')
    axs[0].plot(range(len(trainLossHistory)), trainLossHistory,
                label="train", color='tab:orange')
    axs[0].plot([stepPerEpoch*i for i in range(epochNum)],
                validLossHistory,
                label="valid", color='tab:red')
    axs[0].legend()
    axs[0].set(xticks=[stepPerEpoch*i for i in range(epochNum)],
               xticklabels=list(range(epochNum))
               )

    axs[1].set_xlabel('epoch')
    axs[1].set_ylabel('accuracy')
    axs[1].plot(range(len(trainAccHistory)), trainAccHistory,
                label="train", color='tab:blue')
    axs[1].plot([stepPerEpoch*i for i in range(epochNum)],
                validAccHistory,
                label="valid", color='tab:green')
    axs[1].legend()
    axs[1].set(xticks=[stepPerEpoch*i for i in range(epochNum)],
               xticklabels=list(range(epochNum))
               )
    plt.grid(True)
    plt.savefig(os.path.join(saveDir, title + '.pdf'))


def plot_smooth_loss_acc(saveDir, title="smooth_loss_accuracy"):
    trainLossHistory = np.array(torch.load(os.path.join(saveDir, "trainLossHistory")))
    trainAccHistory = np.array(torch.load(os.path.join(saveDir, "trainAccHistory")))
    validLossHistory = torch.load(os.path.join(saveDir, "validLossHistory"))
    validAccHistory = torch.load(os.path.join(saveDir, "validAccHistory"))

    epochNum = len(validLossHistory)
    stepPerEpoch = len(trainLossHistory)//epochNum

    fig, axs = plt.subplots(2, 1, figsize=(18, 10))
    axs[0].set_xlabel('epoch')
    axs[0].set_ylabel('loss')
    axs[0].plot(range(epochNum),
                [trainLossHistory[stepPerEpoch*i:stepPerEpoch*(i+1)].mean()for i in range(epochNum)],
                label="train", color='tab:orange')
    axs[0].plot(range(epochNum),
                validLossHistory,
                label="valid", color='tab:red')
    axs[0].legend()

    axs[1].set_xlabel('epoch')
    axs[1].set_ylabel('accuracy')
    axs[1].plot(range(epochNum),
                [trainAccHistory[stepPerEpoch*i:stepPerEpoch*(i+1)].mean()for i in range(epochNum)],
                label="train", color='tab:blue')
    axs[1].plot(range(epochNum),
                validAccHistory,
                label="valid", color='tab:green')
    axs[1].legend()
    plt.grid(True)
    plt.savefig(os.path.join(saveDir, title + '.pdf'))


if __name__ == '__main__':
    plot_smooth_loss_acc(r"..\data\experiment\2019-12-31 19.45")

