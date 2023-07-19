
import torch
import torch.nn as nn
import pandas as pd
import argparse

from utils import MyDataset, get_file_list
from torch.utils.data import DataLoader
from tqdm import tqdm

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # print(x)
        # print(x.shape)
        # print(x.dtype)
        output, _ = self.gru(x)
        # print("second layer")
        # print(output.shape)
        output = self.bn(output)
        output = self.fc(output)
        return output

def loss_fn(y_pred, y_true):
    y = torch.cat((y_pred.view(1, -1), y_true.view(1, -1)), dim=0)
    corr = torch.corrcoef(y)[0, 1]
    return -corr


def pred_result(test_dataset, model):
    test_loader = DataLoader(test_dataset, batch_size=1024)
    model.eval()
    test_target = []
    test_output = []
    for input, target in tqdm(test_loader):
        output = model(input)
        test_output.append(output)
        test_target.append(target)
    pred = torch.concat(test_output).squeeze()
    true = torch.concat(test_target).squeeze()
    test_loss = loss_fn(pred, true)
    print(pred.shape)
    print(pred)

    result = pd.DataFrame(columns=['date', 'stock_code', 'pred'])
    result = []
    for i in tqdm(range(len(test_dataset))):
        stock, date = test_dataset.get_info(i)
        # result.loc[i] = [date, stock, pred[i]]
        result.append([date, stock, pred[i].item()])
    result = pd.DataFrame(result, columns=['date', 'stock_code', 'pred'])

    print(test_loss)
    print(result)
    result.to_csv(f"./full_data/result/{date}.csv", index=False)

# python prediction.py --start 2017-01-01 --end 2017-06-31 --model 2016-12-32
# python prediction.py --start 2017-07-01 --end 2017-12-32 --model 2017-06-31
# python prediction.py --start 2018-01-01 --end 2018-06-31 --model 2017-12-32
# python prediction.py --start 2018-07-01 --end 2018-12-32 --model 2018-06-31
# python prediction.py --start 2019-01-01 --end 2019-06-31 --model 2018-12-32
# python prediction.py --start 2019-07-01 --end 2019-12-32 --model 2019-06-31
# python prediction.py --start 2020-01-01 --end 2020-06-31 --model 2019-12-32
# python prediction.py --start 2020-07-01 --end 2020-12-32 --model 2020-06-31
# python prediction.py --start 2021-01-01 --end 2021-06-31 --model 2020-12-32
# python prediction.py --start 2021-07-01 --end 2021-12-32 --model 2021-06-31
# python prediction.py --start 2022-01-01 --end 2022-06-31 --model 2021-12-32
# python prediction.py --start 2022-07-01 --end 2022-12-32 --model 2022-06-31
# python prediction.py --start 2023-01-01 --end 2023-06-31 --model 2022-12-32

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, help="start date", default="2017-01-01")
    parser.add_argument("--end", type=str, help="end date", default="2017-06-31")
    parser.add_argument("--model", type=str, help="choosing the model", default="2016-12-32")
    args = parser.parse_args()


    state_dict = torch.load(f"./full_data/result/{args.model}_model.pt")
    # Create a new instance of the model
    input_size = 1920
    hidden_size = 30
    output_size = 1
    learning_rate = 0.0001
    num_epochs = 50
    batch_size = 1024

    # 创建模型和优化器
    model = GRUModel(input_size, hidden_size, output_size)
    # Load the saved parameters and buffers into the model
    model.load_state_dict(state_dict)

    # Load data
    file_list_init = get_file_list('./full_data/min_datapoint/')
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

    # filelist = ['./data/2020-03-02.txt', './data/2020-03-09.txt', './data/2020-03-16.txt', './data/2020-03-23.txt', './data/2020-03-30.txt', './data/2020-04-07.txt', './data/2020-04-14.txt']
    test_dataset = MyDataset(test_list)

    pred_result(test_dataset, model)
