import pandas as pd
import argparse
import torch.nn.functional as F
import torch
import torch.nn as nn

from utils import single_dataset, get_file_list, loss_fn
from torch.utils.data import DataLoader
from model import GRUModel_serial

# 定义训练函数
def train(model, optimizer, train_loader, device):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # print("shit")
        # print(data.shape)

        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)



# python main.py --end 2020-07-01
# python main.py --end 2016-12-32
# python main.py --end 2017-06-31
# python main.py --end 2017-12-32
# python main.py --end 2018-06-31
# python main.py --end 2018-12-32
# python main.py --end 2019-06-31
# python main.py --end 2019-12-32
# python main.py --end 2020-06-31
# python main.py --end 2020-12-32
# python main.py --end 2021-06-31
# python main.py --end 2021-12-32
# python main.py --end 2022-06-31
# python main.py --end 2022-12-32
# python main.py --end 2023-06-31
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--end", type=str, help="end date", default="John")
    args = parser.parse_args()


    file_list_init = get_file_list('./full_data/min_datapoint/')
    file_list = []
    test_end = args.end # '2020-07-01'

    for i in range (len(file_list_init)):
        if i % 5 == 0 and file_list_init[i][-14:] <= test_end:
            file_list.append(file_list_init[i])

    train_len = int(len(file_list) * 4 / 5)
    train_list = file_list[:train_len]
    test_list = file_list[train_len:]
    print(train_list)
    print(test_list)


    train_dataset = single_dataset(train_list)
    test_dataset = single_dataset(test_list)

    # 设置超参数
    input_size = 6          # six feature in one time step
    hidden_size = 30
    output_size = 1
    learning_rate = 0.0001
    num_epochs = 50
    batch_size = 1024

    # 创建数据集和数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)


    # 创建模型和优化器
    model = GRUModel_serial(input_size, hidden_size, output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    device = torch.device('cuda')
    model.to(device)


    # 训练模型
    best_test_loss = 0
    best_train_loss = 0
    for epoch in range(num_epochs):
        train_loss = train(model, optimizer, train_loader, device)
        if best_train_loss > train_loss: best_train_loss = train_loss
        # print('Epoch: {}, Train Loss: {:.4f}'.format(epoch+1, train_loss))
        # 在每个epoch结束后对测试数据进行预测
        model.eval()
        test_loss = 0
        test_output = []
        test_target = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_output.append(output)
                test_target.append(target)

            pred = torch.concat(test_output).squeeze()
            true = torch.concat(test_target).squeeze()
            test_loss = loss_fn(pred, true)
        if(epoch % 10 == 0):
            y = torch.cat((pred.view(1, -1), true.view(1, -1)), dim=0)
            print(pred)
            print(true)
        print('Epoch: {}, Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch+1, train_loss, test_loss))
        if best_test_loss > test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), f"./full_data/result_min/{test_end}_model.pt")
    
    print(f"best train loss {best_train_loss}, best test loss {best_test_loss}")

    with open("./full_data/result_min/loss.log", 'a') as f:
        f.write(f'\nDate {test_end}')
        f.write(f'\nBest Train Loss: {best_train_loss}')
        f.write(f'\nBest Test Loss: {best_test_loss}')
