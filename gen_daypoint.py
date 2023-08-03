import os
import pandas as pd
import numpy as np
import pickle
import builtins 

from multiprocessing import Manager, Pool
from datetime import datetime, timedelta
from tqdm import tqdm

# return date_list table_list
def read_table(dir):
    file_list = os.listdir(dir)[:600]
    print(file_list)
    table_list = []
    date_list = []
    for file in tqdm(file_list):
        date_list.append(datetime.strptime(file.split('.')[0], "%Y-%m-%d"))
        table_list.append(pd.read_feather(dir + file))
    
    return date_list, table_list

# input numpy shape (x, y), normalize x
def time_norm(table):
    mean = np.mean(table, axis=1).reshape(-1, 1)
    result = table / mean
    return result

# input numpy shape (x, y), normalize y
def zscore_norm(table):
    return (table - np.mean(table, axis=0)) / np.std(table, axis=0)

# generate daypoint using all stock info
# 0-39: feature, 39 close - 49 close: return
# args = [
# date_list: containing all date 
# table_list: containing all table dataframe
# i: the selected date index
# ]
def process_daypoint(args):
    date_list, table_list, i = args
    stock_set = set(table_list[i]["code"])
    # stock datetime    target  open    high    low close   vwap    volume
    # 1     1           1       320     320     320 320     320     320
    basic_info = []
    target = []
    open = []
    high = []
    low = []
    close = []
    vwap = []
    volume = []

    # get the included stock code & group table by code
    j = i - 39
    group_table = []
    while j <= i: # 0-39
        stock_set = stock_set & set(table_list[j]["code"])
        group_table.append(table_list[j].groupby('code'))
        j += 1
    while j <= i + 10: #39-49
        stock_set = stock_set & set(table_list[j]["code"])
        j += 1

    # generate daypoint for each stock
    for stock in stock_set:
        # get basic info [stock, date]
        basic_info.append([stock, date_list[i].strftime('%Y-%m-%d')])

        # get target
        today_close = table_list[i][table_list[i]["code"] == stock]["close"].item()[-1]
        future_close = table_list[i + 10][table_list[i + 10]["code"] == stock]["close"].item()[-1]
        target.append([(future_close - today_close) / today_close])

        # get feature
        result = [[], [], [], [], [], []]
        for iter_table in group_table:
            stock_table = iter_table.get_group(stock).reset_index(drop=True)
            result[0].extend(list(stock_table["open"].tolist()[0]))
            result[1].extend(list(stock_table["high"].tolist()[0]))
            result[2].extend(list(stock_table["low"].tolist()[0]))
            result[3].extend(list(stock_table["close"].tolist()[0]))
            result[4].extend(list(stock_table["vwap"].tolist()[0]))
            tmp_v = list(stock_table["volume"].tolist()[0]) # set volume to 1 if it is 0
            result[5].extend([1 if x == 0 else x for x in tmp_v]) 

        open.append(result[0])
        high.append(result[1])
        low.append(result[2])
        close.append(result[3])
        vwap.append(result[4])
        volume.append(result[5])

    # normalize feature
    volume = time_norm(np.array(volume))
    open = time_norm(np.array(open))
    high = time_norm(np.array(high))
    low = time_norm(np.array(low))
    close = time_norm(np.array(close))
    vwap = time_norm(np.array(vwap))
    basic_info = np.array(basic_info)

    open = zscore_norm(open)
    high = zscore_norm(high)
    low = zscore_norm(low)
    close = zscore_norm(close)
    vwap = zscore_norm(vwap)
    volume = zscore_norm(volume)
    target = zscore_norm(target)

    day_data = np.concatenate([basic_info, target, open, high, low, close, vwap, volume], axis=1)
    # with builtins.open(f"./full_data/day_datapoint/{date_list[i].strftime('%Y-%m-%d')}.pickle", 'wb') as f:
    #     pickle.dump(day_data, f)
    np.savetxt(f"./full_data/day_datapoint/{date_list[i].strftime('%Y-%m-%d')}.txt", day_data, fmt="%s")
    print(date_list[i].strftime('%Y-%m-%d'))

def process_daypoint_week(args):
    week_list, date_list, week_index, dir = args
    date_index = date_list.index(week_list[week_index][-1])
    basic_info = []
    target = []
    open = []
    high = []
    low = []
    close = []
    vwap = []
    volume = []

    # load 40 table for feature and 2 table for target
    j = date_index - 39
    table_list = []
    while j <= date_index:
        table_list.append(pd.read_csv(dir + date_list[j].strftime('%Y-%m-%d') + '.csv'))
        j += 1
    w1_table = pd.read_csv(dir + week_list[week_index + 1][0].strftime('%Y-%m-%d') + '.csv')
    w2_table = pd.read_csv(dir + week_list[week_index + 2][0].strftime('%Y-%m-%d') + '.csv')

    # select which stock to include
    group_table = []
    stock_set = set(table_list[0]["code"])
    for table in table_list:
        stock_set = stock_set & set(table["code"])
        group_table.append(table.groupby('code'))
    stock_set = stock_set & set(w1_table["code"])
    stock_set = stock_set & set(w2_table["code"])

    for stock in (stock_set):
        # get basic info [stock, date]
        basic_info.append([stock, week_list[week_index][0].strftime('%Y-%m-%d')])

        # get target
        W1_close = eval(w1_table[w1_table["code"] == stock]["close"].item())[-1]
        W2_close = eval(w2_table[w2_table["code"] == stock]["close"].item())[-1]
        target.append([(W2_close - W1_close) / W1_close])

        # get feature
        result = [[], [], [], [], [], []]
        for iter_table in group_table:
            stock_table = iter_table.get_group(stock).reset_index(drop=True)
            result[0].extend(eval(stock_table["open"].tolist()[0]))
            result[1].extend(eval(stock_table["high"].tolist()[0]))
            result[2].extend(eval(stock_table["low"].tolist()[0]))
            result[3].extend(eval(stock_table["close"].tolist()[0]))
            result[4].extend(eval(stock_table["vwap"].tolist()[0]))
            tmp_v = eval(stock_table["volume"].tolist()[0]) # set volume to 1 if it is 0
            result[5].extend([1 if x == 0 else x for x in tmp_v])
        
        open.append(result[0])
        high.append(result[1])
        low.append(result[2])
        close.append(result[3])
        vwap.append(result[4])
        volume.append(result[5])

    # normalize feature
    volume = time_norm(np.array(volume))
    open = time_norm(np.array(open))
    high = time_norm(np.array(high))
    low = time_norm(np.array(low))
    close = time_norm(np.array(close))
    vwap = time_norm(np.array(vwap))
    basic_info = np.array(basic_info)

    open = zscore_norm(open)
    high = zscore_norm(high)
    low = zscore_norm(low)
    close = zscore_norm(close)
    vwap = zscore_norm(vwap)
    volume = zscore_norm(volume)
    target = zscore_norm(target)

    day_data = np.concatenate([basic_info, target, open, high, low, close, vwap, volume], axis=1)
    np.savetxt(f"./full_data/week_datapoint/{date_list[date_index].strftime('%Y-%m-%d')}.txt", day_data, fmt="%s")
    print(date_list[date_index].strftime('%Y-%m-%d'))

# generate daypoints using weeks' data. the last day of W_0 is training point,
# W_1-W_2 is target
# This function will not load all table at first. Tables are loaded when needed.
def generate_daypoint_week():
    dir = "./full_data/day_table/"
    file_list = os.listdir(dir)
    date_list = []
    for file in file_list:
        date_list.append(datetime.strptime(file[:-4], "%Y-%m-%d"))

    # arrage date in weeks' fmt
    week_list = []
    week = [date_list[0]]
    for i in range(1, len(date_list)):
        if date_list[i - 1].weekday() >= date_list[i].weekday():
            week_list.append(week)
            week = [date_list[i]]
        elif date_list[i - 1] + timedelta(days=4) <= date_list[i]:
            week_list.append(week)
            week = [date_list[i]]
        else:
            week.append(date_list[i])

    # get the start date of generating, and the start week of generating
    start_date = date_list[39]
    i = 0
    while start_date not in week_list[i]:
        i += 1
    start_week_index = i
    start_date = week_list[i][-1]

    # generate datapoint for each week
    # for i in range(start_week_index, len(week_list) - 2):
    #     process_daypoint_week([week_list, date_list, i, dir])

    args_list = [(week_list, date_list, i, dir) for i in range(start_week_index, len(week_list) - 2)]
    with Pool(processes=6) as pool:
        pool.map(process_daypoint_week, args_list)

# generate day datapoint
def generate_daypoint():
    date_list, table_list = read_table("./full_data/day_table/")
    print(len(date_list))
    print(date_list[39:-10])

    # serial 
    # for i in range(39, len(date_list) - 10):
    #     process_daypoint([date_list, table_list, i])

    args_list = [( date_list, table_list, i) for i in range(39, len(date_list) - 10)]
    with Pool(processes=20) as pool:
        pool.map(process_daypoint, args_list)



if __name__ == "__main__":
    generate_daypoint()

    # generate_daypoint_week()


