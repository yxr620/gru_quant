import os
import pandas as pd

def concate_result(dir):
    dir = "./full_data/result_min/"
    file_list_init = os.listdir(dir)
    file_list = []
    table_list = []
    for file in file_list_init:
        if file[-3:] == 'csv':
            table = pd.read_csv(dir + file)
            table = table.drop(table.columns[0], axis=1)
            table_list.append(table)
            file_list.append(file)
            print(table)


    result = pd.concat(table_list, axis=0)
    result.to_csv(dir + "concate_result.csv", index=False)

if __name__ == "__main__":
    dir = "./full_data/result_min/"
    concate_result(dir)
    # df1 = pd.DataFrame({'A': [1, 2], 'B': [4, 5]})
    # df2 = pd.DataFrame({'A': [7, 8, 9], 'B': [10, 11, 12]})

    # result = pd.concat([df1, df2], axis=0)
    # print(result)
