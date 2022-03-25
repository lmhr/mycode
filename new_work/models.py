import torch
from torch._C import device
import torchvision.models as models
from torch import nn
from torchsummary import summary

def get_models(name, num_classes):
    """
    使用模型名称：
    预测分类的类别
    """
    if name == 'vgg16':
        model = models.vgg16(pretrained=True)
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.classifier._modules['6'] = nn.Linear(4096, 7)#修改模型分类数（不用添加softmax层，nn.CrossEntropyLoss默认加上了softmax）
    elif name == 'resnet18':
        model = models.resnet18(pretrained=True)
        for parameter in model.parameters():
            parameter.requires_grad = False
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif name == 'densenet121':
        model = models.densenet121(pretrained=True)
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.classifier = nn.Linear(1024, 7)
    # set_parameters_require_grad(model, True)
    return model

def set_parameters_require_grad(model, is_fixed):
    #默认parameter.requires_grad = True
    #当采用固定预训练模型参数的方法进行训练时，将预训练模型的参数设置成不需要计算梯度
    if(is_fixed):
        for parameter in model.parameters():
            parameter.requires_grad = False

#模型测试
# torch.cuda.set_device(3)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = get_models('densenet121', 7).to(device)

#打印模型结构
# print(model)
#测试模型-1
# summary(model, (3, 224, 224))
#测试模型-2
# test = torch.rand((1, 3, 224, 224)).to(device)
# print(test.shape)
# print(model(test).shape)
# for name, blk in model.named_children():
#     test = blk(test)
#     print(name, 'output shape: ', test.shape)