import numpy as np
import os
import torch
import torchvision.models as models
from torch import nn
import cv2
from src import mysql

FACE_SIZE = 256
EXPAND_RATE = 1.1

class Report():
    def __init__(self, useCuda):
        # 此模型用于在一帧图像检测人脸位置
        self.face_cascade = cv2.CascadeClassifier('E:/test\Algorithm\src/face_cascade\haarcascade_frontalface_default.xml')

        self.model_extract_face_feature = models.resnet18(pretrained=True)
        self.model_extract_face_feature.fc = nn.Identity()
        self.model_extract_face_feature.eval()
        if useCuda:
            self.model_extract_face_feature.cuda()
        self.useCuda = useCuda
        #try:
        self.db = mysql.MysqlOperator()
        self.face_features = self.db.readData()
        # except:
        #     print("fail to connect to mysql server..")
        #     self.db = None
        #     self.face_features = None

        self.last_most_close_worker_id = -1

    def report(self, img_tensor, wear_hat_confidence,datetime,photo):
        """
        由face_detection中的主循环调用
        如果检测到未带安全帽的工人id与上个记录不同
        则向数据库存入一条新记录
        :param img_tensor: 
        :param wear_hat_confidence: 
        :return: 
        """
        if self.db is None:
            print("Mysql is not available, this record will be abandon")
            return
        print(img_tensor.shape)
        out = self.model_extract_face_feature(img_tensor).detach().cpu().numpy()
        print("wear_hat_confidence:", wear_hat_confidence)
        diff = ((self.face_features - out)**2).sum(1)
        print(diff)
        print(diff.shape)
        most_close_worker_id = int(np.argmin(diff))+1
        print(most_close_worker_id)
        sql = "select * from workers where id=" + str(most_close_worker_id)
        self.db.cur.execute(sql)
        # 执行sql语句
        results = self.db.cur.fetchall()
        name=0
        for row in results:
            name=row[2]
            most_close_worker_id = row[1]
        print(name)
        print(most_close_worker_id)
        if most_close_worker_id != self.last_most_close_worker_id and name!="":
            self.db.insertRecord(wear_hat_confidence, most_close_worker_id,datetime,name,photo)
            self.last_most_close_worker_id = most_close_worker_id

    def incertWorker(self, name, gender, photo):
        if self.db is None:
            print("Mysql is not available, this worker will be abandon")
            return

        # faces是检测到的人脸坐标列表
        faces = self.face_cascade.detectMultiScale(photo, 1.1, 5)

        if len(faces) == 1:
            face = self.chopFace(photo, faces[0])
            # face是人脸区域缩放成(FACE_SIZE, FACE_SIZE)的正方形图像
            face = cv2.resize(face, (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_LINEAR)
        else:
            print("Warning: no face or more than 1 faces detected")
            face = self.chopCenter(photo)

        img_tensor = rep.nparr2tensor(face)
        face_feature = self.model_extract_face_feature(img_tensor)
        face_feature = self.tensor2nparr(face_feature)[0]
        self.db.insertWorker(name, gender, photo, face_feature)

    @staticmethod
    def chopFace(img, pos, expand=True):
        (x, y, w, h) = pos
        if not (img.ndim == 3 and img.shape[2] == 3):
            return img
        centerY = y + h//2
        centerX = x + w//2
        if expand:
            halfSideLength = int((max(w, h)//2) * EXPAND_RATE)
            # 防止放大的区域超出图片边界
            possibleConstraints = (halfSideLength, centerX, centerY, img.shape[0]-centerY, img.shape[1]-centerX)
            halfSideLength = min(possibleConstraints)
        else:
            halfSideLength = (max(w, h) // 2)
        print(img.shape)
        print(halfSideLength)
        return img[centerY-halfSideLength:centerY+halfSideLength,
               centerX-halfSideLength:centerX+halfSideLength, :]

    @staticmethod
    def chopCenter(img):
        if not (img.ndim == 3 and img.shape[2] == 3):
            return img
        halfSideLength = min(img.shape[0], img.shape[1])//2 - 1
        centerY = img.shape[0]//2
        centerX = img.shape[1]//2
        return img[centerY-halfSideLength:centerY+halfSideLength,
               centerX-halfSideLength:centerX+halfSideLength]

    def nparr2tensor(self, nparr):
        if self.useCuda:
            # 神经网络输入张量格式处理
            img_tensor = torch.from_numpy(nparr.astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0).cuda()
        else:
            # 神经网络输入张量格式处理
            img_tensor = torch.from_numpy(nparr.astype(np.float32) / 255).permute(2, 0, 1).unsqueeze(0)
        return img_tensor

    def tensor2nparr(self, tensor):
        if self.useCuda:
            nparr = tensor.detach().cpu().detach().numpy()
        else:
            nparr = tensor.detach().numpy()
        return nparr

if __name__ == '__main__':
    """插入一个工人样例"""
    scale = 0.1
    # print(os.listdir("./"))
    img = cv2.imread("img/face.jpg")
    img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)),
                     interpolation=cv2.INTER_LINEAR)
    cv2.imshow("test", img)
    cv2.waitKey(100)
    rep = Report(False)
    rep.incertWorker("xu", 1, img)
    print(img.shape)


