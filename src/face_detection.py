import os

import cv2
import numpy as np
import torch
from src import resnet
import datetime
from src import report

SAVE_ROOT = "../data/experiment"

def load_CNN_model(saveDir):
    net = resnet.resnet(in_channel=3, num_classes=1)
    print("Loading model", saveDir)
    ckptPath = os.path.join(saveDir, "best_ckpt")
    useCuda = torch.cuda.is_available()
    if useCuda:
        ckpt = torch.load(ckptPath)
    else:
        ckpt =torch.load(ckptPath, map_location='cpu')
    net.load_state_dict(ckpt['model_state_dict'])
    if useCuda:
        net.cuda()
    net.eval()
    return net, useCuda


def face_rec_pc(saveRoot=None, dirId=-1, threshold=0.5):
    """
    此函数中是摄像头识别人脸，判别是否佩戴安全帽，存数据进数据库的主循环
    :param saveRoot: 
    :param dirId: 读取的卷积网络编号，-1表示最后一个，也就是训练阶段最后保存的
    :param threshold: 戴安全帽的概率值小于多少视为触发报告
    :return: 
    """
    if saveRoot is None:
        saveRoot = SAVE_ROOT
    useCuda = torch.cuda.is_available()
    dirName = os.listdir(saveRoot)[dirId]
    saveDir = os.path.join(saveRoot, dirName)

    # 此模型用于二分类人脸图像是否佩戴安全帽，输入人脸部分图像，输出戴安全帽的概率值
    model, useCuda = load_CNN_model(saveDir)
    rep = report.Report(useCuda)

    print("Opening camera..")
    camera = cv2.VideoCapture(0)
    while True:
        read, img = camera.read()
        if not read: continue
        # faces是检测到的人脸坐标列表
        faces = rep.face_cascade.detectMultiScale(img, 1.1, 5)
        # 遍历每一张图像中的人脸
        for (x, y, w, h) in faces:
            # face是人脸区域缩放成96*96的正方形图像
            face = cv2.resize(rep.chopFace(img,(x, y, w, h)), (96, 96), interpolation=cv2.INTER_LINEAR)
            img_tensor = rep.nparr2tensor(face)
            out = model(img_tensor)
            if out < threshold:
                #如果戴安全帽的概率值小于threshold，触发报告，画红框
                img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                dat = datetime.datetime.now()
                rep.report(img_tensor, out.item(),dat,img)
            else:
                #画蓝框
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imshow("camera", img)
        if cv2.waitKey(1000 // 12) & 0xff == ord('q'):
            break
    cv2.destroyAllWindows()


