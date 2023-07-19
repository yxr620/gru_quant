import torch
import pickle
import numpy as np
import os

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

def get_feature(datepoint):
    # print(datepoint)
    info = [datepoint[0], datepoint[1]]
    target = [datepoint[2].astype(np.double)]
    feature = datepoint[3:].astype(np.double)
    return info, feature, target

def get_file_list(dir):
    file_list = os.listdir(dir)
    for i in range(len(file_list)):
        file_list[i] = dir + file_list[i]
    return file_list

def is_trading_day(date) -> bool:
    try:
        df = d.mink(date.strftime("%Y-%m-%d"), columns=['code'])
    except FileNotFoundError as e:
        return False
    # if df.empty(): return False
    return True

# 定义自定义数据集类
class MyDataset(Dataset):
    def __init__(self, file_list):
        self.target = []
        self.feature = []
        self.info = []

        for file in tqdm(file_list):
            day_data = np.loadtxt(file, dtype=str)
            for i in range(day_data.shape[0]):
                info, feature, target = get_feature(day_data[i])
                self.info.append(info)
                self.feature.append(feature)
                self.target.append(target)
        
        self.feature = np.array(self.feature)
        self.target = np.array(self.target)
        # the torch type must match the model type
        self.feature = torch.tensor(self.feature, dtype=torch.float32)
        self.target = torch.tensor(self.target, dtype=torch.float32)
        print(self.feature.shape)
        # print(len(self.target))
        # print(len(self.feature))

    def __getitem__(self, index):
        x = self.feature[index]
        y = self.target[index]
        return x, y
    
    def __len__(self):
        return len(self.feature)
    
    def get_info(self, index):
        return self.info[index]
    

    
