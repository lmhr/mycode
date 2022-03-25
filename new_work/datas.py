import torchvision
from torchvision import transforms
import os
import sys
import torch

def get_datas(data_dir, resize, batch_size, train_transfrom, test_transfrom):
    """
    输入：路径，图片size，读取批次，训练集采用的数据增强,测试集的数据增强
    """
    train_images = torchvision.datasets.ImageFolder(os.path.join(data_dir, "train"), train_transfrom)
    test_images = torchvision.datasets.ImageFolder(os.path.join(data_dir, "validation"), test_transfrom)
    if sys.platform.startswith('win'):#设置多进程读取在Windows系统中，num_workers参数建议设为0，在Linux系统则不需担心。
        num_workers = 0  # 0表示不用额外的进程来加速读取数据
    else:
        num_workers = 16
    train_loader = torch.utils.data.DataLoader(train_images, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_images, batch_size=batch_size, num_workers=num_workers)#shuffle打乱数据
    return train_loader, test_loader