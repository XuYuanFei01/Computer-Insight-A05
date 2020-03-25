import  xml.dom.minidom
from xml.dom.minidom import parse
import os
import cv2
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
from sklearn import model_selection
from tqdm import tqdm
from multiprocessing import Process


DATA_PATH = "../data"
IMG_PATH = "../data/JPEGImages"
ANNOTATION_PATH = "../data/Annotations"
TRANSFORM = transforms.Compose([transforms.ToPILImage(),
                                      transforms.Resize(128),  # 重置图像分辨率
                                      transforms.RandomResizedCrop(96),  # 随机裁剪
                                      transforms.RandomHorizontalFlip(),  # 以概率p水平翻转
                                      transforms.ToTensor(),
                                      ])
NUM_XML_PER_PROCESS = 1000
NUM_IMG_PER_FILE = 100
TOTAL_IMG_NUM = 7583


def demoReadXML(name="000348"):
    domTree = parse(os.path.join(ANNOTATION_PATH, name+".xml"))
    img = cv2.imread(os.path.join(IMG_PATH, name+".jpg"))
    # 文档根元素
    rootNode = domTree.documentElement
    # print(rootNode.nodeName)
    objects = rootNode.getElementsByTagName("object")
    for obj in objects:
        name = obj.getElementsByTagName("name")[0].childNodes[0].data

        bndboxNode = obj.getElementsByTagName("bndbox")[0]
        xmin = int(bndboxNode.getElementsByTagName("xmin")[0].childNodes[0].data)
        xmax = int(bndboxNode.getElementsByTagName("xmax")[0].childNodes[0].data)
        ymin = int(bndboxNode.getElementsByTagName("ymin")[0].childNodes[0].data)
        ymax = int(bndboxNode.getElementsByTagName("ymax")[0].childNodes[0].data)
        difficult = int(obj.getElementsByTagName("difficult")[0].childNodes[0].data)

        if difficult: continue
        if name == "person":
            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        elif name == "hat":
            img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    cv2.imshow("img", img)
    cv2.waitKey(0)


class WholeDataset(Dataset):
    def __init__(self, transform=TRANSFORM):
        self.fileList = os.listdir(ANNOTATION_PATH)
        try:
            self.objNum = np.load(os.path.join(DATA_PATH, "objNum.npy"))
            print("successfully load objNum")
        except FileNotFoundError:
            self.objNum = np.zeros(len(self.fileList), dtype=np.int64)
            self.get_obj_num_in_xmls()

        self.len = self.objNum.sum()
        self.objNumCumSum = np.cumsum(self.objNum)
        self.transform = transform

    def get_obj_num_in_xmls(self):
        print("began counting obj num in xmls..")
        for i, xml in tqdm(enumerate(self.fileList)):
            fileName = xml.rstrip(".xml")
            domTree = parse(os.path.join(ANNOTATION_PATH, fileName + ".xml"))
            img = cv2.imread(os.path.join(IMG_PATH, fileName + ".jpg"))
            # 文档根元素
            rootNode = domTree.documentElement
            self.objNum[i] = len(rootNode.getElementsByTagName("object"))

        np.save(os.path.join(DATA_PATH, "objNum"), self.objNum)


    def __getitem__(self, index):
        fileId = np.searchsorted(self.objNumCumSum, index)
        fileName = self.fileList[fileId].rstrip(".xml")
        domTree = parse(os.path.join(ANNOTATION_PATH, fileName + ".xml"))
        img = cv2.imread(os.path.join(IMG_PATH, fileName + ".jpg"))
        # 文档根元素
        rootNode = domTree.documentElement
        objects = rootNode.getElementsByTagName("object")
        localId = index - (self.objNumCumSum[fileId-1] if fileId > 0 else 0) -1
        assert self.objNum[fileId] == len(objects)
        obj = objects[localId]
        name = obj.getElementsByTagName("name")[0].childNodes[0].data
        difficult = int(obj.getElementsByTagName("difficult")[0].childNodes[0].data)

        bndboxNode = obj.getElementsByTagName("bndbox")[0]
        xmin = int(bndboxNode.getElementsByTagName("xmin")[0].childNodes[0].data)
        xmax = int(bndboxNode.getElementsByTagName("xmax")[0].childNodes[0].data)
        ymin = int(bndboxNode.getElementsByTagName("ymin")[0].childNodes[0].data)
        ymax = int(bndboxNode.getElementsByTagName("ymax")[0].childNodes[0].data)
        objImg = self.transform(img[ymin:ymax,xmin:xmax])


        if name == "hat":
            label = torch.tensor([1])
        else:
            label = torch.tensor([0])

        return objImg, label

    def __len__(self):
        return self.len

class SplitDataset(Dataset):
    def __init__(self, wholeDataset, referenceArr):
        self.d = wholeDataset
        self.r = referenceArr

    def __getitem__(self, index):
        return self.d[self.r[index]]

    def __len__(self):
        return len(self.r)


def get_train_test_set(test_size=0.05, showPostiveSamplePercentage=False):
    allDataset = WholeDataset()
    choices = np.random.rand(len(allDataset))
    trainReference = []
    testReference = []
    for i, prob in enumerate(choices):
        if prob > test_size:
            trainReference.append(i)
        else:
            testReference.append(i)
    # print(trainReference)
    trainset = SplitDataset(allDataset, trainReference)
    testset = SplitDataset(allDataset, testReference)
    assert len(trainset)+len(testset) == len(allDataset)
    print("successfully load trainset of size {}".format(len(trainset)))
    print("successfully load testset of size {}".format(len(testset)))
    numPostiveSample = 0
    if showPostiveSamplePercentage:
        for _, label in testset:
            numPostiveSample += int(label == torch.tensor([1]))
        print("percentage of postive sample is about {}%".format(100*numPostiveSample/len(testset)))
    return trainset, testset


if __name__ == '__main__':
    demoReadXML("000203")
    #demoReadXML("000670")
    #demoReadXML("PartA_01746")
    trainset, testset = get_train_test_set()
    print(trainset[0][0].mean())

