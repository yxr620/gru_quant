import os
import pandas as pd
import numpy as np


from multiprocessing import Manager, Pool
from scipy.stats import zscore
from datetime import datetime, timedelta
from tqdm import tqdm

# return date_list table_list
def read_table(dir):
    file_list = os.listdir(dir)
    print(file_list)
    table_list = []
    date_list = []
    for file in file_list:
        date_list.append(datetime.strptime(file[:-4], "%Y-%m-%d"))
        table_list.append(pd.read_csv(dir + file))
    
    return date_list, table_list


# input numpy shape (x, y), normalize x
def time_norm(table):
    mean = np.mean(table, axis=1).reshape(-1, 1)
    result = table / mean
    return result

# input numpy shape (x, y), normalize y
def zscore_norm(table):
    return (table - np.mean(table, axis=0)) / np.std(table, axis=0)

# generate datapoint using all stock info
def process_datapoint_all(args):
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

    # print(date_list[i].strftime('%Y-%m-%d'))
    for stock in stock_set:
        # judge if the stock satisfy the requirements
        satisfy = 1
        j = i - 19
        while j <= i + 10:
            if stock not in set(table_list[j]["code"]): satisfy = 0
            j += 1
        if satisfy == 0: continue

        basic_info.append([stock, date_list[i].strftime('%Y-%m-%d')])

        # get target
        today_close = eval(table_list[i][table_list[i]["code"] == stock]["close"].item())[-1]
        future_close = eval(table_list[i + 10][table_list[i + 10]["code"] == stock]["close"].item())[-1]
        target.append([(future_close - today_close) / today_close])

        # get feature
        j = i - 19
        result = [[], [], [], [], [], []]
        while j <= i:
            stock_table = table_list[j][table_list[j]["code"] == stock]
            result[0].extend(eval(stock_table["open"].tolist()[0]))
            result[1].extend(eval(stock_table["high"].tolist()[0]))
            result[2].extend(eval(stock_table["low"].tolist()[0]))
            result[3].extend(eval(stock_table["close"].tolist()[0]))
            result[4].extend(eval(stock_table["vwap"].tolist()[0]))
            tmp_v = eval(stock_table["volume"].tolist()[0])
            result[5].extend([1 if x == 0 else x for x in tmp_v])
            # if all(element == 0 for element in tmp_v):
            #     result[5].extend([1] * len(tmp_v))
            # else: result[5].extend(tmp_v)

            j += 1
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

    # print(volume)
    # np.savetxt("result_tmp.txt", volume)
    # exit()

    open = zscore_norm(open)
    high = zscore_norm(high)
    low = zscore_norm(low)
    close = zscore_norm(close)
    vwap = zscore_norm(vwap)
    volume = zscore_norm(volume)
    target = zscore_norm(target)

    day_data = np.concatenate([basic_info, target, open, high, low, close, vwap, volume], axis=1)
    np.savetxt(f"./full_data/min_datapoint/{date_list[i].strftime('%Y-%m-%d')}.txt", day_data, fmt="%s")
    print(date_list[i].strftime('%Y-%m-%d'))

def generate_datapoint():
    date_list, table_list = read_table("./full_data/min_table/")

    # serial 
    # for i in range(20, len(date_list) - 10):
    #     process_datapoint_all([date_list, table_list, i])

    # parallel
    args_list = [( date_list, table_list, i) for i in range(20, len(date_list) - 10)]
    with Pool(processes=4) as pool:
        pool.map(process_datapoint_all, args_list)

if __name__ == "__main__":

    # using all stock to generate datapoint
    generate_datapoint()



