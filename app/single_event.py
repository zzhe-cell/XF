from datetime import datetime
import pandas as pd
#paramList = {'start_time', 'end_time', 'threshold', 'time_column', 'num_column' }
def get_time_window(start_time, end_time, data):
    start_time = datetime.strptime(start_time, '%Y-%m-%d')
    end_time = datetime.strptime(end_time, '%Y-%m-%d')
    data[time_column] = data[time_column].astype('datetime64')
    window_data = data.loc[(data[time_column] >= start_time) & (data[time_column] <= end_time)]
    window_data['来访时间'] = window_data['来访时间'].astype(str)
    # window_data = data.loc[(data['来访时间'] >= start_time) & (data['来访时间'] <= end_time)]
    return window_data
df = get_time_window(start_time, end_time, df)
df = df[df[num_column] >= threshold]
df = df.reset_index(drop=True)


