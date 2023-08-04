import torch
import numpy as np
import pandas as pd
import os
import multiprocessing

from multiprocessing import Pool
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

# delete stock whose adjfactor is empty, resulting in nan for open, high, low, close
def fix_table():
    min_dir = "./full_data/min_table/2019-12-13.feather"
    day_dir = "./full_data/day_table/2019-12-13.feather"
    stock = "000043.SZ"
    min_table = pd.read_feather(min_dir)
    day_table = pd.read_feather(day_dir)

    min_table = min_table[min_table['code'] != stock].reset_index(drop=True)
    day_table = day_table[day_table['code'] != stock].reset_index(drop=True)

    min_table.to_feather(min_dir)
    day_table.to_feather(day_dir)

# the file name of required datapoint. Only the name needed not the entire dir
class single_dataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        self.target = []
        self.feature = []
        self.info = []

        with Pool(processes=15) as pool:
            results = pool.map(self.load_file, self.file_list)

        for info_list, feature_list, target_list in results:
            self.info.extend(info_list)
            self.feature.extend(feature_list)
            self.target.extend(target_list)

        self.feature = np.array(self.feature)
        self.target = np.array(self.target)
        # the torch type must match the model type
        self.feature = torch.tensor(self.feature, dtype=torch.float32)
        self.target = torch.tensor(self.target, dtype=torch.float32)
        print(self.feature.shape)

    def load_file(self, file):
        day_data = np.loadtxt(file, dtype=str)
        info_list, feature_list, target_list = [], [], []

        for i in range(day_data.shape[0]):
            info, feature, target = get_min_feature(day_data[i])
            info_list.append(info)
            feature_list.append(feature)
            target_list.append(target)

        print(file)
        return info_list, feature_list, target_list

    def __getitem__(self, index):
        x = self.feature[index]
        y = self.target[index]
        return x, y
    
    def __len__(self):
        return len(self.feature)
    
    def get_info(self, index):
        return self.info[index]


class double_dataset(Dataset):
    def __init__(self, file_list):
        self.target = []
        self.feature1 = []
        self.feature2 = []
        self.info = []

        self.min_dir = "./full_data/min_datapoint/"
        self.day_dir = "./full_data/day_datapoint/"

        with Pool(processes=15) as pool:
            results = pool.map(self.process_file, file_list)
            
        for feature1, feature2, target, info in results:
            self.feature1.extend(feature1)
            self.feature2.extend(feature2)
            self.target.extend(target)
            self.info.extend(info)
            # print(info)

        self.feature1 = np.array(self.feature1)
        self.feature2 = np.array(self.feature2)
        self.target = np.array(self.target)
        self.feature1 = torch.tensor(self.feature1, dtype=torch.float32)
        self.feature2 = torch.tensor(self.feature2, dtype=torch.float32)
        self.target = torch.tensor(self.target, dtype=torch.float32)
        print(self.feature1.shape)
        print(self.feature2.shape)
        # print(self.info)

    def process_file(self, file):
        print(file)
        min_data = np.loadtxt(self.min_dir + file, dtype=str)
        day_data = np.loadtxt(self.day_dir + file, dtype=str)

        min_stocks = min_data[:,0]  
        day_stocks = day_data[:,0]
        common_stocks = np.intersect1d(min_stocks, day_stocks)
        common_mask_min = np.isin(min_data[:,0], common_stocks)
        common_mask_day = np.isin(day_data[:,0], common_stocks)
        common_min_data = min_data[common_mask_min]
        common_day_data = day_data[common_mask_day]

        feature1 = []
        target = []
        info_result = []
        feature2 = []
        for i in range(common_min_data.shape[0]):
            info, feature, target_val = get_min_feature(common_min_data[i])
            feature1.append(feature)
            target.append(target_val)
            info_result.append(info)

        for i in range(common_day_data.shape[0]):
            info, feature, target_val = get_day_feature(common_day_data[i])
            feature2.append(feature)

        return feature1, feature2, target, info_result

    def __len__(self):
        return len(self.feature1)
      
    def __getitem__(self, index):
        return self.feature1[index], self.feature2[index], self.target[index]

    def get_info(self, index):
        return self.info[index]

