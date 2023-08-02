import torch
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

# get min datapoint [320 * 6]
def get_min_feature(datapoint):
    info = [datapoint[0], datapoint[1]]
    target = [datapoint[2].astype(np.double)]
    feature = datapoint[3:].astype(np.double).reshape(6, 320).T
    return info, feature, target
# get day datapoint [40 * 6]
def get_day_feature(datapoint):
    info = [datapoint[0], datapoint[1]]
    target = [datapoint[2].astype(np.double)]
    feature = datapoint[3:].astype(np.double).reshape(6, 40).T
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

def loss_fn(y_pred, y_true):
    y = torch.cat((y_pred.view(1, -1), y_true.view(1, -1)), dim=0)
    corr = torch.corrcoef(y)[0, 1]
    return -corr


# the file name of required datapoint. Only the name needed not the entire dir
class single_dataset(Dataset):
    def __init__(self, file_list):
        self.target = []
        self.feature = []
        self.info = []

        for file in tqdm(file_list):
            day_data = np.loadtxt(file, dtype=str)
            for i in range(day_data.shape[0]):
                info, feature, target = get_min_feature(day_data[i])
                self.info.append(info)
                self.feature.append(feature)
                self.target.append(target)
        
        self.feature = np.array(self.feature)
        self.target = np.array(self.target)
        # the torch type must match the model type
        self.feature = torch.tensor(self.feature, dtype=torch.float32)
        self.target = torch.tensor(self.target, dtype=torch.float32)
        print(self.feature.shape)


    def __getitem__(self, index):
        x = self.feature[index]
        y = self.target[index]
        return x, y
    
    def __len__(self):
        return len(self.feature)
    
    def get_info(self, index):
        return self.info[index]


# 自定义数据集
class double_dataset(Dataset):
    def __init__(self, file_list):
        self.target = []
        self.feature1 = []
        self.feature2 = []
        self.info = []

        min_dir = "./full_data/min_datapoint/"
        day_dir = "./full_data/day_datapoint/"
        for file in tqdm(file_list):
            min_data = np.loadtxt(min_dir + file, dtype=str)
            day_data = np.loadtxt(day_dir + file, dtype=str)

            # get the same stock in both min and day data
            min_stocks = min_data[:,0]  
            day_stocks = day_data[:,0]
            common_stocks = np.intersect1d(min_stocks, day_stocks)
            common_mask_min = np.isin(min_data[:,0], common_stocks)
            common_mask_day = np.isin(day_data[:,0], common_stocks)
            common_min_data = min_data[common_mask_min]
            common_day_data = day_data[common_mask_day]
            # print(common_day_data)
            for i in range(common_min_data.shape[0]):
                info, feature, target = get_min_feature(common_min_data[i])
                self.info.append(info)
                self.feature1.append(feature)
                self.target.append(target)
            for i in range(common_day_data.shape[0]):
                info, feature, target = get_day_feature(common_day_data[i])
                self.feature2.append(feature)   

        self.feature1 = np.array(self.feature1)
        self.feature2 = np.array(self.feature2)
        self.target = np.array(self.target)
        # the torch type must match the model type
        self.feature1 = torch.tensor(self.feature1, dtype=torch.float32)
        self.feature2 = torch.tensor(self.feature2, dtype=torch.float32)
        self.target = torch.tensor(self.target, dtype=torch.float32)
        print(self.feature1.shape)
        print(self.feature2.shape)
        # print(len(self.target))
        # print(len(self.feature))

    def __len__(self):
        return len(self.feature1)
      
    def __getitem__(self, index):
        return self.feature1[index], self.feature2[index], self.target[index]
    def get_info(self, index):
        return self.info[index]

