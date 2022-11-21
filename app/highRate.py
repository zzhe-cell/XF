import time
from datetime import datetime

import pandas as pd
time_column =  '来访时间'
n = 6
file_path = '../data/信访事项统计表明细_xx.csv'
df = pd.read_csv(file_path, encoding='gbk')
paramsList = {'time_column', 'n'}
def get_time_window(start_time, end_time, data):
    start_time = datetime.strptime(start_time, '%Y-%m-%d')
    end_time = datetime.strptime(end_time, '%Y-%m-%d')
    data[time_column] = data[time_column].astype('datetime64')
    window_data = data.loc[(data[time_column] >= start_time) & (data[time_column] <= end_time)]
    window_data[time_column] = window_data[time_column].astype(str)
    # window_data = data.loc[(data['来访时间'] >= start_time) & (data['来访时间'] <= end_time)]
    return window_data
def get_frequent_event(data, n):
    end_year = time.localtime(time.time())[0]
    start_year = end_year
    end_month = time.localtime(time.time())[1]
    start_month = end_month
    for i in range(n):
        if start_month == 1:
            start_year -= 1
            month = 12
        else:
            start_month -= 1
    start_time = str(start_year) + '-' + str(start_month) + '-1'
    end_time = str(end_year) + '-' + str(end_month) + '-1'
    data = data.sort_values([time_column])
    window_data = get_time_window(start_time, end_time, data)
    return window_data

df = get_frequent_event(df, n)
print(df)