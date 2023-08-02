import os
import pandas as pd
import numpy as np
import argparse

from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager, Pool
from tqdm import tqdm

# read all table from dir
# return date_list(containing all date) table_list(containing all table dataframe)
def read_table(dir):
    file_list = os.listdir(dir)
    print(file_list)
    table_list = []
    date_list = []
    for file in tqdm(file_list):
        date_list.append(datetime.strptime(file[:-4], "%Y-%m-%d"))
        table_list.append(pd.read_csv(dir + file))
    
    print("finish load")
    return date_list, table_list

# read raw tables from feather file and save the raw table to disks
def save_adj_table(args):
    api, date, file = args

    new_table = pd.read_feather(file)[
        ['code', 'datetime', 'open', 'high', 
        'low', 'close', 'volume', 'amount']
    ]
    stock_list = list(set(new_table["code"]))
    adjfactor = api.wsd(stock_list, ["adjfactor"], date.strftime('%Y-%m-%d'), date.strftime('%Y-%m-%d'))

    new_table.set_index('code', inplace=True)
    new_table['open'] = new_table['open'] * adjfactor['adjfactor']
    new_table['high'] = new_table['high'] * adjfactor['adjfactor']
    new_table['low'] = new_table['low'] * adjfactor['adjfactor']
    new_table['close'] = new_table['close'] * adjfactor['adjfactor']
    new_table['amount'] = new_table['amount'] * adjfactor['adjfactor']
    new_table.reset_index(inplace=True)
    new_table.to_csv(f"full_data/adj_table/{date.strftime('%Y-%m-%d')}.csv", index=False)
    print(date)


# generate the feature for 15min data
# return list of (6, 16)
def get_feature_sep(table):
    stock_df = table
    result = [[], [], [], [], [], []]
    for index, row in stock_df[["open","high","low","close","volume","amount"]].iterrows():
        if index % 15 == 0:
            open = row["open"]
            high = row["high"]
            low = row["low"]
            vwap = row["amount"]
            volume = row["volume"]
        else:
            if row["high"] > high: high = row["high"]
            if row["low"] < low: low = row["low"]
            volume += row["volume"]
            vwap += row["amount"]

        if index % 15 == 14: # visit the last line of 15 min
            close = row["close"]
            result[0].append(open)
            result[1].append(high)
            result[2].append(low)
            result[3].append(close)
            if volume == 0:
                result[4].append(open)
                result[5].append(0)
            else:
                result[4].append(vwap / volume)
                result[5].append(volume)

    return result

# generate the feature for 15min data
# return list of (6, 1)
def get_feature_day(table):
    # open high low close vwap volume
    result = [[], [], [], [], [], []]
    
    open = table["open"].iloc[0]
    high = table["high"].max()
    low = table["low"].min()
    if low == 0:
        low = table["low"].sort_values().iloc[1]
    close = table["close"].iloc[-1]
    if close == 0:
        close = table["close"].iloc[-2]
    vwap = table["amount"].sum()
    volume = table["volume"].sum()
    result[0].append(open)
    result[1].append(high)
    result[2].append(low)
    result[3].append(close)

    if volume == 0:
        result[4].append(open)
        result[5].append(0)
    else:
        result[4].append(vwap / volume)
        result[5].append(volume)

    return result

# save table ... without delisting stock, save to ./full_data/min_table. table in the following format:
# code open high low close vwap volume
# str  list ...
def process_table_all(args):
    date, table = args
    stock_set = set(table["code"])

    open = []
    high = []
    low = []
    close = []
    vwap = []
    volume = []
    stock_list = []
    table_group = table.groupby('code')
    for stock in stock_set:
        result = get_feature_sep(table_group.get_group(stock).reset_index(drop=True))
        if all(element == 0 for element in result[5]):
            continue
        stock_list.append(stock)
        open.append(result[0])
        high.append(result[1])
        low.append(result[2])
        close.append(result[3])
        vwap.append(result[4])
        volume.append(result[5])

    stock_table = pd.DataFrame(list(stock_list), columns=["code"])
    stock_table["open"] = open
    stock_table["high"] = high
    stock_table["low"] = low
    stock_table["close"] = close
    stock_table["vwap"] = vwap
    stock_table["volume"] = volume


    stock_table.to_csv(f"./full_data/min_table/{date.strftime('%Y-%m-%d')}.csv", index=False)
    print(date.strftime('%Y-%m-%d'))

# save table for day data in the following format:
# code open high low close vwap volume
# str  list ...
def process_table_day(args):
    date, table = args
    stock_set = set(table["code"])

    open = []
    high = []
    low = []
    close = []
    vwap = []
    volume = []
    stock_list = []
    table_group = table.groupby('code')
    for stock in stock_set:
        result = get_feature_day(table_group.get_group(stock).reset_index(drop=True))
        # print(result)
        if all(element == 0 for element in result[5]):
            continue

        stock_list.append(stock)
        open.append(result[0])
        high.append(result[1])
        low.append(result[2])
        close.append(result[3])
        vwap.append(result[4])
        volume.append(result[5])

    stock_table = pd.DataFrame(list(stock_list), columns=["code"])
    stock_table["open"] = open
    stock_table["high"] = high
    stock_table["low"] = low
    stock_table["close"] = close
    stock_table["vwap"] = vwap
    stock_table["volume"] = volume

    # print(stock_table)
    # exit()
    stock_table.to_csv(f"./full_data/day_table/{date.strftime('%Y-%m-%d')}.csv", index=False)
    print(date.strftime('%Y-%m-%d'))

# read adj table and generate down sampled table 
def generate_all_table():
    date_list, table_list = read_table("./full_data/adj_table/")

    # for i in range(len(date_list)):
    #     process_table_all([date_list[i], table_list[i]])

    args_list = [(date_list[i], table_list[i]) for i in range(len(date_list))]
    with Pool(processes=4) as pool:
        pool.map(process_table_all, args_list)

# read adj table and generate day table
def generate_day_table():
    date_list, table_list = read_table("./full_data/adj_table/")

    # for i in range(len(date_list)):
    #     process_table_day([date_list[i], table_list[i]])

    args_list = [(date_list[i], table_list[i]) for i in range(len(date_list))]
    with Pool(processes=4) as pool:
        pool.map(process_table_day, args_list)


# using adj factor to adjust raw table and save it
def adjtable():
    from PyLocalData import d
    d.start()
    dir = "./full_data/1min/"
    file_list = os.listdir(dir)
    date_list = []
    for i in range(len(file_list)):
        date_list.append(datetime.strptime(file_list[i][:-8], "%Y%m%d"))
        file_list[i] = dir + file_list[i]

    args_list = [(d, date_list, file_list, i) for i in range(len(date_list))]

    # for i in range(len(args_list)):
    #     save_adj_table(args_list[i])

    args_list = [(d, date_list[i], file_list[i]) for i in range(len(date_list))]
    with Pool(processes=4) as pool:
        pool.map(save_adj_table, args_list)

# python table_pro.py --adj --min --day
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adj", action='store_true', help="processing adj table")
    parser.add_argument("--min", action='store_true', help="processing adj table to min table")
    parser.add_argument("--day", action='store_true', help="processing adj table to day table")
    args = parser.parse_args()

    print(args)
    # read feather table, adj factor and save the table to file
    if args.adj:
        adjtable()

    # generate table with no normalization ...
    # table contain all stock without delisting
    if args.min:
        generate_all_table()

    # generate day table
    if args.day:
        generate_day_table()



