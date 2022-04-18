import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from models.model import Trilinear_GCN
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from dataloader import affectnet_csv
from utils.loss_function import CenterLoss
from utils.loss_function import AffinityLoss
from sklearn.metrics import confusion_matrix
from utils.plot_cinfusion_matrix import plot_confusion_matrix
import cv2


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sava_path = "C:/Users/USER/Desktop/pythonProject1/models/affectNet"

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def joint_loss(restnet_outs, node_outs, centor_outs, targets_gpu):

    gamma1 = 1.32
    gamma2 = 1.0
    gamma3 = 1.0
    gamma4 = 0.1

    criterion = nn.CrossEntropyLoss().to(device)
    center_loss = CenterLoss(feat_dim=2048)
    affinity_loss = AffinityLoss(device='cuda', num_class=8, feat_dim=512)
    loss = gamma2 * criterion(restnet_outs, targets_gpu) \
           + gamma3 * criterion(node_outs, targets_gpu) + gamma4 * affinity_loss(centor_outs, targets_gpu)

    return loss

def train(batch_size, epochs, lr, criterion, mini_batch):

    model = Trilinear_GCN(num_classes=8)
    model.to(device)
    best_acc = 0.0

    gamma1 = 1.1
    gamma2 = 1.2
    gamma3 = 1.0
    gamma4 = 1.1

    mean_acc = np.zeros(0)

    # loss function part
    criterion = nn.CrossEntropyLoss().to(device)
    affinity_loss = AffinityLoss(device='cuda', num_class=8, feat_dim=512)

    # Data transforms
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.RandomAffine(20, scale=(0.8, 1), translate=(0.2, 0.2)),
        ], p=0.7),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(),
    ])

    train_set = affectnet_csv.AffectNet(aff_path="C:/Users/USER/Desktop/AffectNetData", phase="train", transform=data_transforms,)
    train_loader = DataLoader(train_set, batch_size=batch_size,shuffle=True, num_workers=0, pin_memory=True, drop_last=True, )

    step_size = int(train_loader.__len__()) * 21

    # optimizer part
    params = list(model.parameters()) + list(affinity_loss.parameters())
    optimizer = optim.Adam(params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0

        for param_group in optimizer.param_groups:
            print('Learning rate: ', param_group['lr'])

        for i, data in enumerate(tqdm(train_loader)):

            inputs, targets = data
            targets_gpu = targets.to(device)

            resnet_outs, centor_outs = model(inputs.cuda())
            loss = gamma2 * criterion(resnet_outs, targets_gpu) \
            + gamma4 * affinity_loss(centor_outs, targets_gpu)

            acc = accuracy(resnet_outs, targets_gpu)
            mean_acc = np.append(mean_acc, float(acc[0]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            if (i % mini_batch) == mini_batch-1:
                print("\n")
                print("{} epoch's loss : {:.4f}".format(epoch + 1, running_loss/mini_batch))
                print("accuarcy = {:.4f}".format(mean_acc.mean()))
                # print("target : {}, prediction : {}".format(targets[0], graph_outs[0]))
                mean_acc = np.zeros(0)

        torch.save(model, sava_path + "/last_affectnet_model.pt")
        score = test(model=model, batch_size=batch_size)
        if (score > best_acc):
            torch.save(model, sava_path + "/best_affectnet_model.pt")
            best_acc = score


def test(model, batch_size = 64):
    model.eval()

    # Data transforms
    data_transforms_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    test_loss = 0.0
    criterrion = nn.CrossEntropyLoss().to(device)

    test_dataset = affectnet_csv.AffectNet(aff_path="C:/Users/USER/Desktop/AffectNetData", phase="val", transform=data_transforms_val)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    div = test_loader.__len__()
    mean_acc = np.zeros(0)
    res_mean_acc = np.zeros(0)

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            images, target = data
            output, res_out= model(images.cuda())

            acc = accuracy(output.cpu().detach(), target)
            # acc_res = accuracy(res_out.cpu().detach(), target)

            loss = criterrion(output, target.to(device))

            test_loss += loss.item()

            mean_acc = np.append(mean_acc, acc)
            # res_mean_acc = np.append(res_mean_acc, acc_res)

    print("test loss : {:.4f}".format(test_loss / div))
    print("Test Accuracy : {:.4f}".format(mean_acc.mean()))
    # print("Resnet feature Test Accuracy : {:.4f}".format(res_mean_acc.mean()))

    return mean_acc.mean()


def evaluate_model(web_cam, model=None, batch_size = 64, save_confusion_mat=False):

    if model is None:
        model = torch.load(sava_path + "/best_affectnet_model.pt")

    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    # pre_labels = []
    # gt_labels = []
    #
    # test_dataset = affectnet_csv.AffectNet(aff_path="C:/Users/USER/Desktop/AffectNetData", phase="val", transform=data_transforms_val)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    # mean_acc = np.zeros(0)

    web_cam = torch.from_numpy(web_cam)
    web_cam = web_cam.permute(2, 0, 1)
    web_cam = data_transforms_val(web_cam)
    web_cam = web_cam.unsqueeze(dim=0)

    model.eval()
    output, _ = model(web_cam)
    # with torch.no_grad():
    #     for i, data in enumerate(test_loader):
    #         images, target = data
    #         output, _ = model(images.cuda())
    #
    #         _, predicts = torch.max(output, 1)
    #
    #         pre_labels += predicts.cpu().tolist()
    #         gt_labels += target.cpu().tolist()
    #
    #         acc = accuracy(output.cpu().detach(), target)
    #
    #         mean_acc = np.append(mean_acc, acc)
    #
    #     if save_confusion_mat:
    #         cm = confusion_matrix(gt_labels, pre_labels)
    #         cm = np.array(cm)
    #         labels_name = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "angry", "contempt"]
    #         plot_confusion_matrix(cm, labels_name, 'Affectnet-8', "62.43")

    # print("Evaluation Accuracy : {:.4f}".format(mean_acc.mean()))
    return output


def main():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if cap.isOpened():
        ret, a = cap.read()
        emotions = ["netural", "happiness", "sadness", "surprise", "fear", "disgust", "anger"]

        while ret:
            ret, a = cap.read()
            gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
            results = face_cascade.detectMultiScale(gray,  # 입력 이미지
                                                    scaleFactor=1.1,  # 이미지 피라미드 스케일 factor
                                                    minNeighbors=5,  # 인접 객체 최소 거리 픽셀
                                                    minSize=(20, 20)  # 탐지 객체 최소 크기
                                                    )

            for box in results:
                x, y, w, h = box
                cv2.rectangle(a, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
                face_image = a.copy()
                square = face_image[y:y+h, x:x+w]
                if w > 100:
                    square = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
                    resnet_outs = evaluate_model(web_cam=square, batch_size=1, save_confusion_mat=True)
                    max_index = np.argmax(resnet_outs)
                    print(emotions[max_index])

            cv2.imshow('Emotion Recognition', a)

            if cv2.waitKey(1) & 0xFF == 27:
                break
            # 종료 커맨드.

    cap.release()
    cv2.destroyAllWindows()

# if __name__ == "__main__":
    # main()

print(cv2.__version__)