import cv2
import os
import numpy as np
import torch
from resnet import resnet

SAVE_ROOT = "../data/experiment"


def load_CNN_model(saveDir):
    net = resnet(in_channel=3, num_classes=1)
    print("Loading model", saveDir)
    ckptPath = os.path.join(saveDir, "best_ckpt")
    ckpt = torch.load(ckptPath)
    assert torch.cuda.is_available() == ckpt["useCuda"]
    net.load_state_dict(ckpt['model_state_dict'])
    if ckpt["useCuda"]:
        net.cuda()
    net.eval()
    return net


def face_rec(saveRoot=None, dirId=-1):
    if saveRoot is None:
        saveRoot = SAVE_ROOT
    useCuda = torch.cuda.is_available()
    dirName = os.listdir(saveRoot)[dirId]
    saveDir = os.path.join(saveRoot, dirName)

    model = load_CNN_model(saveDir)

    print("Opening camera..")
    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(r'face_cascade/haarcascade_frontalface_default.xml')
    while True:
        read, img = camera.read()
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        for (x, y, w, h) in faces:
            face = cv2.resize(img[x:x+w,y:y+h,:], (96, 96), interpolation=cv2.INTER_LINEAR)
            if useCuda:
                img_tensor = torch.from_numpy(face.astype(np.float32)/255).permute(2, 0, 1).unsqueeze(0).cuda()
            else:
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            out = model(img_tensor)
            print(out)
            if out<0.5:
                img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            else:
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv2.imshow("camera", img)
        if cv2.waitKey(1000 // 12) & 0xff == ord('q'):
            break
    cv2.destroyAllWindows()


