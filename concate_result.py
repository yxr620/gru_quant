import os
import pandas as pd

def concate_result(dir):
    file_list_init = os.listdir(dir)
    file_list = []
    table_list = []
    for file in file_list_init:
        if file[-3:] == 'csv':
            table = pd.read_csv(dir + file)
            # table = table.drop(table.columns[0], axis=1)
            table_list.append(table)
            file_list.append(file)
            print(table)


    result = pd.concat(table_list, axis=0).reset_index(drop=True)
    result.to_feather(dir + "concate_result.feather")

if __name__ == "__main__":
    dir = "./full_data/result_min/"
    dir_day = "./full_data/result_day/"
    dir_double = "./full_data/result_double/"
    concate_result(dir_double)

