
import torch
import torch.nn as nn
import pandas as pd
import argparse
import os
import numpy as np

from utils import single_dataset, get_file_list, double_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import *
from scipy.stats import pearsonr

def loss_fn(y_pred, y_true):
    y = torch.cat((y_pred.view(1, -1), y_true.view(1, -1)), dim=0)
    corr = torch.corrcoef(y)[0, 1]
    return -corr

def calculate_ic(group):
    ic, _ = pearsonr(group['pred'], group['target'])
    return ic

# calculate IC, rank IC, etc
def cal_ic(result, dir):
    grouped = result.groupby('date')
    result['pred_rank'] = result.groupby('date')['pred'].rank()
    result['target_rank'] = result.groupby('date')['target'].rank()
    ic_values = grouped.apply(lambda x: pearsonr(x['pred'], x['target'])[0])
    rank_ic_values = grouped.apply(lambda x: pearsonr(x['pred_rank'], x['target_rank'])[0])

    # IC mean etc
    ic_mean = ic_values.mean()
    ic_std = ic_values.std()
    ic_ir = ic_mean / ic_std
    postive_ic = len(ic_values[ic_values > 0]) / len(ic_values)

    f = open(dir + "test.log", 'a')

    f.write(f"\nIC Mean: {ic_mean:.5f}")
    f.write(f"\nIC Std: {ic_std:.5f}")
    f.write(f"\nRank IC: {rank_ic_values.mean():.5f}")
    f.write(f"\nIC_IR: {ic_ir:.5f}")
    f.write(f"\npostive_ic: {postive_ic:.5f}")


def pred_result_double(test_dataset, model, device, dir):
    test_loader = DataLoader(test_dataset, batch_size=1024)
    model.eval()
    test_target = []
    test_output = []
    with torch.no_grad():
        for input1, input2, target in tqdm(test_loader):
            input1, input2, target = input1.to(device), input2.to(device), target.to(device)
            output = model(input1, input2)
            test_output.append(output)
            test_target.append(target)
    pred = torch.concat(test_output).squeeze()
    true = torch.concat(test_target).squeeze()
    test_loss = loss_fn(pred, true)
    print(pred.shape)
    print(pred)

    result = []
    for i in tqdm(range(len(test_dataset))):
        stock, date = test_dataset.get_info(i)
        _, _, target = test_dataset[i]
        result.append([date, stock, pred[i].item(), float(target)])
    result = pd.DataFrame(result, columns=['date', 'stock_code', 'pred', 'target'])

    print(test_loss)
    print(result)
    cal_ic(result, dir)
    result[['date', 'stock_code', 'pred']].to_csv(dir + f"{date}.csv", index=False)
    return test_loss


def pred_result(test_dataset, model, device):
    test_loader = DataLoader(test_dataset, batch_size=256)
    model.eval()
    test_target = []
    test_output = []

    # get pred and label
    with torch.no_grad():
        for batch_idx, (input, target) in tqdm(enumerate(test_loader)):
            input = input.to(device)
            output = model(input)
            output = output.cpu()
            test_output.append(output)
            test_target.append(target)

    pred = torch.concat(test_output).squeeze()
    true = torch.concat(test_target).squeeze()
    test_loss = loss_fn(pred, true)
    print(pred.shape)
    print(pred)

    result = []
    for i in tqdm(range(len(test_dataset))):
        stock, date = test_dataset.get_info(i)
        _, target = test_dataset[i]
        # result.loc[i] = [date, stock, pred[i]]
        result.append([date, stock, pred[i].item(), float(target)])
    result = pd.DataFrame(result, columns=['date', 'stock_code', 'pred', 'target'])

    print(test_loss)
    print(result)

    cal_ic(result, "./full_data/result_min/")
    result[['date', 'stock_code', 'pred']].to_csv(f"./full_data/result_min/{date}.csv", index=False)
    return test_loss


# python prediction.py --start 2020-06-01 --end 2020-06-31 --model 2020-07-01 --type 1

# python prediction.py --start 2017-01-01 --end 2017-06-31 --model 2016-12-32 --type 1 --device cuda
# python prediction.py --start 2017-07-01 --end 2017-12-32 --model 2017-06-31 --type 1 --device cuda
# python prediction.py --start 2018-01-01 --end 2018-06-31 --model 2017-12-32 --type 1 --device cuda
# python prediction.py --start 2018-07-01 --end 2018-12-32 --model 2018-06-31 --type 1 --device cuda
# python prediction.py --start 2019-01-01 --end 2019-06-31 --model 2018-12-32 --type 1 --device cuda
# python prediction.py --start 2019-07-01 --end 2019-12-32 --model 2019-06-31 --type 1 --device cuda
# python prediction.py --start 2020-01-01 --end 2020-06-31 --model 2019-12-32 --type 1 --device cuda
# python prediction.py --start 2020-07-01 --end 2020-12-32 --model 2020-06-31 --type 1 --device cuda
# python prediction.py --start 2021-01-01 --end 2021-06-31 --model 2020-12-32 --type 1 --device cuda
# python prediction.py --start 2021-07-01 --end 2021-12-32 --model 2021-06-31 --type 1 --device cuda
# python prediction.py --start 2022-01-01 --end 2022-06-31 --model 2021-12-32 --type 1 --device cuda
# python prediction.py --start 2022-07-01 --end 2022-12-32 --model 2022-06-31 --type 1 --device cuda
# python prediction.py --start 2023-01-01 --end 2023-06-31 --model 2022-12-32 --type 1 --device cuda

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, help="start date", default="2017-01-01")
    parser.add_argument("--end", type=str, help="end date", default="2017-06-31")
    parser.add_argument("--model", type=str, help="choosing the model, 0 for single 15min model, 1 for 15min+1D model, 2 for two step training model", default="2016-12-32")
    parser.add_argument("--type", type=str, help="select which model is used in the report", default="0")
    parser.add_argument("--device", type=str, help="specify which device to use: cuda cpu", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.type == '0':
        state_dict = torch.load(f"./full_data/result_min/{args.model}_model.pt")
    elif args.type == '1':
        state_dict = torch.load(f"./full_data/result_day/{args.model}_model.pt")
    elif args.type == '2':
        state_dict = torch.load(f"./full_data/result_double/{args.model}_model.pt")
    else:
        print("ERROR model type")
        exit()
    # Create a new instance of the model
    input_size1 = 6
    input_size2 = 6
    hidden_size = 30
    output_size = 1
    num_layers = 1
    learning_rate = 0.0001
    num_epochs = 50
    batch_size = 1024

    # Load data and model para
    if args.type == "0":
        model = LSTMModel_serial(input_size1, hidden_size, num_layers, output_size)
        file_list_init = get_file_list('./full_data/min_datapoint/')
    if args.type == "1":
        model = LSTMModel_double(input_size1=input_size1, input_size2=input_size2, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
        file_list_init = os.listdir('./full_data/min_datapoint/')
    if args.type == '2':
        day_model = LSTMModel_serial(input_size2, hidden_size, num_layers=num_layers, output_size=output_size)
        model = LSTMModel_sep(day_model, input_size1, hidden_size, num_layers=num_layers, output_size=output_size)
        file_list_init = os.listdir('./full_data/min_datapoint/')
    model.load_state_dict(state_dict)
    model.to(device)

    file_list = []
    for i in range(len(file_list_init)):
        if i % 1 == 0:
            file_list.append(file_list_init[i])

    train_end = args.start
    test_end = args.end
    train_list = []
    test_list = []
    for file in file_list:
        if file[-14:] < train_end: train_list.append(file)
        elif file[-14:] < test_end: test_list.append(file)

    print(test_list)
    if args.type == "0":
        with open("./full_data/result_min/test.log", 'a') as f:
            f.write(f'\nDate {test_end}')
        test_dataset = single_dataset(test_list)
        test_loss = pred_result(test_dataset, model, device)
    if args.type == "1":
        with open("./full_data/result_day/test.log", 'a') as f:
            f.write(f'\nDate {test_end}')
        test_dataset = double_dataset(test_list)
        test_loss = pred_result_double(test_dataset, model, device, "./full_data/result_day/")
    if args.type == "2":
        with open("./full_data/result_double/test.log", 'a') as f:
            f.write(f'\nDate {test_end}')
        test_dataset = double_dataset(test_list)
        test_loss = pred_result_double(test_dataset, model, device, "./full_data/result_double/")


