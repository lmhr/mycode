import os
import csv
import numpy as np
import pandas as pd
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class imageDataset(Dataset):

    def __init__(self, file_path):
        self.file_path = file_path
        # 获取路径下所有的图片名称，必须保证路径内没有图片以外的数据
        self.image_list = os.listdir(file_path)
        # 将PIL的Image转为Tensor
        self.transforms = transforms.Compose([
                                          transforms.Resize(224),
                                          transforms.ToTensor()
                                          ])

    def __getitem__(self, index):
         # 根据index获取图片完整路径
        image_path = os.path.join(self.file_path, self.image_list[index])
        image = Image.open(image_path).convert("RGB")
        return self.transforms(image), self.image_list[index]

    def __len__(self):
        return len(self.image_list)

    def get_imagelist(self):
        return self.image_list

if __name__ == "__main__":
    from models import get_models
    from tqdm import tqdm
    print("start prediction:")
    #数据导入
    batch_size = 512
    test_Dataset = imageDataset(file_path="Datasets/test")
    test_loader = torch.utils.data.DataLoader(test_Dataset, batch_size=batch_size, num_workers=16)
    #获得文件模型
    file_name = test_Dataset.get_imagelist()
    #获得模型
    torch.cuda.set_device(2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = 'densenet121'
    net = get_models(name=model_name, num_classes=7).to(device)
    net.load_state_dict(torch.load("DENSENET121.pt", map_location=device))
    net.eval()
    #获得图片预测类型
    prediction=[]
    emotions_all = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    count = 0
    for x,name in tqdm(test_loader):
        for i in net(x.to(device)).argmax(dim=1):
            prediction.append(emotions_all[int(i)])
    
    #写入预测文件
    df1 = pd.DataFrame(file_name)
    df2 = pd.DataFrame(prediction)
    dataframe = pd.concat([df1, df2],axis=1 ,ignore_index=True)
    dataframe.to_csv('prediction.csv', header=None, index=False)