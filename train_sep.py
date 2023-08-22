import torch
import torch.nn as nn
import argparse
import os

from torch.utils.data import Dataset, DataLoader
from utils import double_dataset, loss_fn
from model import LSTMModel_serial, GRUModel_serial, SepModel, LSTMModel_sep

# training using day data
def train_day(model, optimizer, train_loader, device):
    model.train()
    train_loss = 0
    for batch_idx, (x1, x2, target) in enumerate(train_loader):
        x1, x2, target = x1.to(device), x2.to(device), target.to(device)
        optimizer.zero_grad()
        # print("shit")
        # print(data.shape)

        output = model(x2)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)

# training using 15min data with frozen day model
# training using both x1 and x2
def train(model, optimizer, train_loader, device):
    model.train()
    train_loss = 0
    for batch_idx, (x1, x2, target) in enumerate(train_loader):
        x1, x2, target = x1.to(device), x2.to(device), target.to(device)
        optimizer.zero_grad()
        # print("shit")
        # print(data.shape)

        output = model(x1, x2)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)

# python train_sep.py --end 2020-07-01
# python train_sep.py --end 2016-12-32
# python train_sep.py --end 2017-06-31
# python train_sep.py --end 2017-12-32
# python train_sep.py --end 2018-06-31
# python train_sep.py --end 2018-12-32
# python train_sep.py --end 2019-06-31
# python train_sep.py --end 2019-12-32
# python train_sep.py --end 2020-06-31
# python train_sep.py --end 2020-12-32
# python train_sep.py --end 2021-06-31
# python train_sep.py --end 2021-12-32
# python train_sep.py --end 2022-06-31
# python train_sep.py --end 2022-12-32
# python train_sep.py --end 2023-06-31
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--end", type=str, help="end date", default="John")
    args = parser.parse_args()

    min_file = os.listdir('./full_data/min_datapoint/')
    day_file = os.listdir('./full_data/day_datapoint/')
    file_list = []
    test_end = args.end

    for i in range (len(day_file)):
        if i % 5 == 0 and day_file[i][-14:] <= test_end:
            file_list.append(day_file[i])
    file_list = file_list[:-2] # delete the last two weeks

    for file in file_list:
        if file not in min_file:
            print(f"file {file} not in min_file")
            exit()
    train_len = int(len(file_list) * 4 / 5)
    train_list = file_list[:train_len]
    test_list = file_list[train_len:]
    print(train_list)
    print(test_list)

    train_dataset = double_dataset(train_list)
    test_dataset = double_dataset(test_list)

    # 设置超参数
    input_size1 = 6
    input_size2 = 6
    hidden_size = 30
    num_layers = 1
    output_size = 1
    learning_rate = 0.0001
    num_epochs = 50
    batch_size = 1024

    # 创建数据集和数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Instantiate model
    day_model = LSTMModel_serial(input_size2, hidden_size, num_layers=num_layers, output_size=output_size)
    optimizer = torch.optim.Adam(day_model.parameters(), lr=1e-3)
    device = torch.device('cuda')
    day_model.to(device)
    print(day_model)

    # train day model
    best_test_loss = 100
    best_train_loss = 100
    for epoch in range(num_epochs):
        train_loss = train_day(day_model, optimizer, train_loader, device)
        if best_train_loss > train_loss: best_train_loss = train_loss
        # print('Epoch: {}, Train Loss: {:.4f}'.format(epoch+1, train_loss))
        # 在每个epoch结束后对测试数据进行预测
        day_model.eval()
        test_output = []
        test_target = []
        with torch.no_grad():
            for x1, x2, target in test_loader:
                x1, x2, target = x1.to(device), x2.to(device), target.to(device)
                output = day_model(x2)
                test_output.append(output)
                test_target.append(target)
            pred = torch.concat(test_output).squeeze()
            true = torch.concat(test_target).squeeze()
            test_loss = loss_fn(pred, true)
        if(epoch % 10 == 0):
            print(pred)
            print(true)
        print('Epoch: {}, Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch+1, train_loss, test_loss))
        if best_test_loss > test_loss:
            best_test_loss = test_loss
            best_day_model = day_model

    with open("./full_data/result_double/loss.log", 'a') as f:
        f.write(f'\nDate {test_end}')
        f.write(f'\nBest day model Train Loss: {best_train_loss:.5f}')
        f.write(f'\nBest day model Test Loss: {best_test_loss:.5f}')

    # train hybrid model
    hybrid_model = LSTMModel_sep(best_day_model, input_size1, hidden_size, num_layers=num_layers, output_size=output_size)
    optimizer = torch.optim.Adam(hybrid_model.parameters(), lr=learning_rate)
    hybrid_model.to(device)
    print(hybrid_model)
    best_test_loss = 100
    best_train_loss = 100
    for epoch in range(num_epochs):
        train_loss = train(hybrid_model, optimizer, train_loader, device)
        if best_train_loss > train_loss: best_train_loss = train_loss
        # print('Epoch: {}, Train Loss: {:.4f}'.format(epoch+1, train_loss))
        # 在每个epoch结束后对测试数据进行预测
        hybrid_model.eval()
        test_output = []
        test_target = []
        with torch.no_grad():
            for x1, x2, target in test_loader:
                x1, x2, target = x1.to(device), x2.to(device), target.to(device)
                output = hybrid_model(x1, x2)
                test_output.append(output)
                test_target.append(target)
            pred = torch.concat(test_output).squeeze()
            true = torch.concat(test_target).squeeze()
            test_loss = loss_fn(pred, true)
        if(epoch % 10 == 0):
            print(pred)
            print(true)
        print('Epoch: {}, Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch+1, train_loss, test_loss))
        if best_test_loss > test_loss:
            best_test_loss = test_loss
            torch.save(hybrid_model.state_dict(), f"./full_data/result_double/{test_end}_model.pt")

    with open("./full_data/result_double/loss.log", 'a') as f:
        f.write(f'\nBest hybrid model Train Loss: {best_train_loss:.5f}')
        f.write(f'\nBest hybrid model Test Loss: {best_test_loss:.5f}')

