#torch自带VGG不加数据增强
from torch import nn, optim
import torch
import torch.nn.functional as F
import time
import torchvision
import sys
import os
from torchvision import transforms,models
from tqdm import tqdm
import copy
from models import get_models
from datas import get_datas
from torch.utils.tensorboard import SummaryWriter
from begin_train import b_train

#相关参数设置
print("torch自带DENSENET加数据增强")
model_name = 'densenet121'
resize, batch_size = 224, 512
PATH = 'Datasets'#读取路径
num_classes = 7#几分类
lr, num_epochs, weight_decay= 10**-3, 500, 0.1#学习率，迭代次数,权重衰减
#导入模型并部署到设备CPU或者GPU
torch.cuda.set_device(2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = get_models(name=model_name, num_classes=num_classes).to(device)#获得模型
print("training on ", device)
#采用的优化器和损失函数
optimizer = torch.optim.Adam(net.parameters(), lr=lr,weight_decay=weight_decay)
loss = torch.nn.CrossEntropyLoss()#交叉熵损失函数
#描绘下降曲线
train_writer = SummaryWriter('logs/densenet121/train')
test_writer = SummaryWriter('logs/densenet121/test')
#数据增强
train_transfrom = transforms.Compose([
            transforms.ColorJitter(brightness=0.5, contrast=0.5),#随机调整亮度和对比度
            transforms.Resize(resize),
            transforms.RandomResizedCrop(resize/2),#随机裁剪
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),#水平旋转
            transforms.Resize(resize),
            transforms.ToTensor()#转化为tensor
        ])
test_transfrom = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor()#转化为tensor
        ])
#运行流程
train_loader, test_loader = get_datas(PATH, resize, batch_size, train_transfrom, test_transfrom)#获得数据
best_models = b_train(num_epochs,train_loader,test_loader,device,net,loss,optimizer,train_writer,test_writer)#训练
torch.save(best_models,"best_models.pt")